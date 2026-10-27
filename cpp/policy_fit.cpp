// Stream tree dumps, reconstruct placements, and fit a linear policy evaluator.
#include "Eval.hpp"
#include "Util/json.hpp"
#include "engine/Utility.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Json = nlohmann::json;
namespace fs = std::filesystem;
constexpr size_t FEATURES = 43;
using Features = std::array<double, FEATURES>;
static const std::array<std::string, FEATURES> names = {
    "top_half", "top_quarter", "low", "cavity_cells", "cavity_cells_sq",
    "overhangs", "overhangs_sq", "covered_cells", "covered_cells_sq",
    "bumpiness", "bumpiness_sq", "height", "row_transitions", "counting",
    "perfect_clear", "wasted_t",
    "well_columns.0", "well_columns.1", "well_columns.2", "well_columns.3", "well_columns.4",
    "well_columns.5", "well_columns.6", "well_columns.7", "well_columns.8", "well_columns.9",
    "clears.0", "clears.1", "clears.2", "clears.3", "clears.4",
    "tspins.0", "tspins.1", "tspins.2", "tspins.3",
    "back_to_back", "b2b_clear", "combo_garbage", "move_time", "well_depth",
    "mini_tspins.0", "mini_tspins.1", "mini_tspins.2"
};
static constexpr Features scales = {
    1,1,1,8,64,8,64,32,256,16,128,20,64,1,1,1,
    1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1, 1,1,1,1, 1,1,8,40,8,1,1,1
};

struct Action {
    std::array<float, FEATURES> x{};
    double target = 0, prior = 0, old_eval = 0;
};
struct Example {
    double weight = 0;
    std::vector<Action> actions;
};

static double coefficient(const Json& config, const std::string& name) {
    const auto dot = name.find('.');
    return dot == std::string::npos ? config.at(name).get<double>()
        : config.at(name.substr(0, dot)).at(std::stoi(name.substr(dot + 1))).get<double>();
}
static void setCoefficient(Json& config, const std::string& name, double value) {
    const auto dot = name.find('.');
    if (dot == std::string::npos) config[name] = value;
    else config[name.substr(0, dot)][std::stoi(name.substr(dot + 1))] = value;
}

static Features features(const Game& parent, const Piece& piece, int well_cap) {
    Game after = parent;
    const bool held = piece.type != parent.current_piece.type;
    after.place_piece(piece);
    const int lines = after.board.clearLines();
    if (lines > 4) throw std::runtime_error("placement clears more than four lines");
    const auto spin = std::holds_alternative<TetrioS1>(after.mode) && piece.type != PieceType::T
        ? spinType::null : piece.spin;
    const bool full = spin == spinType::normal;
    const bool pc = after.board.is_empty();
    const int attack = after.damage_sent(lines, spin, pc);
    const auto& board = after.board;
    const int height = Eval::height_features(board).second;
    const auto [holes, overhangs] = Eval::cavities_overhangs(board);
    const auto [covered, covered_sq] = Eval::n_covered_cells(board);
    int bump = 0, bump_sq = 0;
    for (size_t x = 1; x < Board::width; ++x) {
        const int d = std::abs(std::countl_zero(board.board[x]) - std::countl_zero(board.board[x-1]));
        bump += d;
        bump_sq += d*d;
    }
    Features f{};
    f[0] = height > 10; f[1] = height > 15; f[2] = height <= 4;
    f[3] = holes; f[4] = holes*holes; f[5] = overhangs; f[6] = overhangs*overhangs;
    f[7] = covered; f[8] = covered_sq; f[9] = bump; f[10] = bump_sq;
    f[11] = height; f[12] = Eval::get_row_transitions(board); f[13] = Eval::ct4(board);
    f[14] = pc; f[15] = piece.type == PieceType::T && !full;
    const int well = Eval::well_position(board);
    f[16 + well] = 1;
    f[26 + lines] = 1;
    if (full && lines < 4) f[31 + lines] = 1;
    f[35] = after.b2b > 0;
    f[36] = lines > 0 && (lines == 4 || spin != spinType::null) && parent.b2b > 0;
    f[37] = lines > 0 ? attack : 0;
    static constexpr int rotations[] = {0,1,2,1};
    f[38] = std::abs(int(piece.position.x)-4) + rotations[int(piece.rotation)] + held + (lines > 0 ? 40 : 0);
    f[39] = Eval::well_depth(board, well, well_cap);
    if (spin == spinType::mini && lines < 3) f[40 + lines] = 1;
    return f;
}

struct Reader {
    uint64_t checked_actions = 0, skipped_nodes = 0;
    double max_reconstruction_error = 0;
    Json teacher;

    template<class Consume>
    void read(const fs::path& path, Consume consume) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot read " + path.string());
        std::string line;
        if (!std::getline(in, line)) throw std::runtime_error("empty dump " + path.string());
        const Json header = Json::parse(line);
        if (header.at("version") != 1 || header.at("kind") != "header")
            throw std::runtime_error("unsupported dump format");
        const Json config = header.at("config");
        if (teacher.is_null()) teacher = config;
        if (config != teacher) throw std::runtime_error("mixed teacher configurations");
        if (config.at("eval") != "nana" || config.at("jeopardy") != 0)
            throw std::runtime_error("this fit requires nana eval with zero jeopardy");
        for (const auto& key : {"tslot", "belief_weights"})
            for (const auto& weight : config.at(key))
                if (weight != 0) throw std::runtime_error("this fit requires zero slot/belief weights");
        Features old{};
        for (size_t k = 0; k < FEATURES; ++k) old[k] = coefficient(config, names[k]);
        while (std::getline(in, line)) {
            const Json node = Json::parse(line);
            double count = 0;
            for (const auto& a : node.at("actions")) count += a.at("N").get<double>();
            if (count <= 0 || node.at("actions").size() < 2) { ++skipped_nodes; continue; }
            Game parent;
            parent.board.board = node.at("board").get<std::array<uint32_t, Board::width>>();
            parent.current_piece = Piece(PieceType(node.at("current").get<int>()));
            if (!node.at("hold").is_null()) parent.hold = PieceType(node.at("hold").get<int>());
            for (size_t i = 0; i < parent.queue.size(); ++i)
                parent.queue[i] = PieceType(node.at("queue").at(i).get<int>());
            parent.b2b = node.at("b2b").get<u16>();
            parent.combo = node.at("combo").get<u16>();
            if (header.at("rules") == "tetrio") parent.mode = TetrioS1();
            else if (header.at("rules") == "botris") parent.mode = Botris();
            else throw std::runtime_error("unknown rules");
            Example example;
            example.weight = node.at("N").get<double>();
            example.actions.reserve(node.at("actions").size());
            for (const auto& a : node.at("actions")) {
                Piece piece(PieceType(a.at("type").get<int>()), RotationDirection(a.at("rotation").get<int>()),
                    Coord{a.at("x").get<i8>(), a.at("y").get<i8>()}, spinType(a.at("spin").get<int>()));
                if (Shaktris::Utility::collides(parent.board, piece))
                    throw std::runtime_error("saved placement collides: " + path.string());
                const auto raw = features(parent, piece, config.at("max_well_depth").get<int>());
                double reconstructed = 1;
                Action action;
                for (size_t k = 0; k < FEATURES; ++k) {
                    action.x[k] = float(raw[k] / scales[k]);
                    reconstructed += old[k]*raw[k]/10000;
                }
                action.target = a.at("N").get<double>() / count;
                action.prior = a.at("prior").get<double>();
                action.old_eval = a.at("eval").get<double>();
                const double error = std::abs(reconstructed-action.old_eval);
                max_reconstruction_error = std::max(max_reconstruction_error, error);
                if (error > 2e-5) throw std::runtime_error("eval reconstruction mismatch: " + std::to_string(error));
                ++checked_actions;
                example.actions.push_back(action);
            }
            consume(std::move(example));
        }
        if (in.bad()) throw std::runtime_error("read failed: " + path.string());
    }
};

static double score(const Features& w, const Action& a) {
    double s = 0;
    for (size_t k = 0; k < FEATURES; ++k) s += w[k]*a.x[k];
    return s;
}

struct Metrics {
    uint64_t nodes = 0, actions = 0;
    double weight = 0, ce = 0, entropy = 0, baseline_ce = 0, uniform_ce = 0;
    double top1 = 0, baseline_top1 = 0;
    void add(const Example& ex, const Features& w) {
        std::vector<double> logits;
        for (const auto& a : ex.actions) logits.push_back(score(w,a));
        const double best = *std::max_element(logits.begin(), logits.end());
        double sum = 0;
        for (double s : logits) sum += std::exp(s-best);
        const double logz = best + std::log(sum);
        double loss = 0, h = 0, old = 0, highest = 0;
        size_t pick = std::max_element(logits.begin(),logits.end())-logits.begin(), oldpick = 0;
        for (size_t i = 0; i < ex.actions.size(); ++i) {
            const auto& a = ex.actions[i];
            loss += a.target*(logz-logits[i]);
            if (a.target > 0) h -= a.target*std::log(a.target);
            old -= a.target*std::log(std::max(a.prior,1e-30));
            highest = std::max(highest,a.target);
            if (a.prior > ex.actions[oldpick].prior) oldpick=i;
        }
        ++nodes; actions += ex.actions.size(); weight += ex.weight;
        ce += ex.weight*loss; entropy += ex.weight*h; baseline_ce += ex.weight*old;
        uniform_ce += ex.weight*std::log(double(ex.actions.size()));
        top1 += ex.weight*(ex.actions[pick].target >= highest-1e-12);
        baseline_top1 += ex.weight*(ex.actions[oldpick].target >= highest-1e-12);
    }
    Json json() const {
        if (!weight) throw std::runtime_error("no usable examples");
        return {{"nodes",nodes},{"actions",actions},{"node_N_sum",weight},
            {"cross_entropy",ce/weight},{"KL",(ce-entropy)/weight},
            {"target_entropy",entropy/weight},{"baseline_cross_entropy",baseline_ce/weight},
            {"baseline_KL",(baseline_ce-entropy)/weight},{"uniform_cross_entropy",uniform_ce/weight},
            {"top1_agreement",top1/weight},{"baseline_top1_agreement",baseline_top1/weight}};
    }
};

struct Adam {
    Features w{}, m{}, v{}, gradient{};
    double batch_weight = 0;
    size_t batch_count = 0, updates = 0;
    void flush(double rate) {
        if (!batch_count) return;
        ++updates;
        for (size_t k = 0; k < FEATURES; ++k) {
            const double g = gradient[k]/batch_weight + 1e-5*w[k];
            m[k] = .9*m[k]+.1*g; v[k] = .999*v[k]+.001*g*g;
            w[k] -= rate*(m[k]/(1-std::pow(.9,double(updates)))) /
                (std::sqrt(v[k]/(1-std::pow(.999,double(updates))))+1e-8);
            if (!std::isfinite(w[k])) throw std::runtime_error("nonfinite model weight");
        }
        gradient = {}; batch_weight = 0; batch_count = 0;
    }
    void add(const Example& ex, double rate) {
        std::vector<double> p;
        for (const auto& a : ex.actions) p.push_back(score(w,a));
        const double best = *std::max_element(p.begin(),p.end());
        double sum = 0;
        for (auto& value : p) { value=std::exp(value-best); sum+=value; }
        for (size_t i = 0; i < p.size(); ++i) {
            const double delta = ex.weight*(p[i]/sum-ex.actions[i].target);
            for (size_t k = 0; k < FEATURES; ++k) gradient[k] += delta*ex.actions[i].x[k];
        }
        batch_weight += ex.weight;
        if (++batch_count == 128) flush(rate);
    }
};

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) throw std::runtime_error("usage: nana_policy_fit DUMP_DIR OUTPUT_DIR [EPOCHS=3]");
        const int epochs = argc == 4 ? std::stoi(argv[3]) : 3;
        if (epochs < 1) throw std::runtime_error("epochs must be positive");
        const fs::path output = argv[2];
        if (fs::exists(output)) throw std::runtime_error("output directory already exists");
        std::vector<fs::path> train, validation;
        for (const auto& entry : fs::directory_iterator(argv[1])) {
            if (entry.path().extension() != ".jsonl") continue;
            const auto stem = entry.path().stem().string();
            if (!stem.starts_with("tree-")) continue;
            const auto seed = std::stoull(stem.substr(5));
            (seed % 10 == 0 ? validation : train).push_back(entry.path());
        }
        if (train.empty() || validation.empty()) throw std::runtime_error("need training and seed%10==0 validation games");
        std::sort(train.begin(),train.end()); std::sort(validation.begin(),validation.end());
        fs::create_directories(output);
        const auto start = std::chrono::steady_clock::now();
        Reader reader;
        Adam model;
        std::mt19937_64 rng(20261028);
        Json report{{"objective","node-N-weighted visit-distribution cross entropy"},
            {"train_games",train.size()},{"validation_games",validation.size()},
            {"validation_split","seed % 10 == 0"},{"feature_count",FEATURES},
            {"shuffle_buffer_nodes",2048},{"epochs",Json::array()}};
        Features best_weights{};
        double best_loss = std::numeric_limits<double>::infinity();
        auto save = [&](int epoch, const Metrics& metrics) {
            const auto result = metrics.json();
            report["epochs"].push_back({{"epoch",epoch},{"validation",result}});
            const double loss = result.at("cross_entropy");
            if (loss < best_loss) {
                best_loss=loss; best_weights=model.w; report["best_epoch"]=epoch;
            }
            Json eval_weights = Json::object();
            Features raw{};
            for (size_t k=0; k<FEATURES; ++k) {
                raw[k]=best_weights[k]/scales[k];
                setCoefficient(eval_weights,names[k],10000*raw[k]);
            }
            report["checked_actions"]=reader.checked_actions;
            report["max_reconstruction_error"]=reader.max_reconstruction_error;
            report["skipped_nodes"]=reader.skipped_nodes;
            report["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            Json artifact{{"version",1},{"purpose","policy prior only"},{"feature_names",names},
                {"feature_scales",scales},{"scaled_weights",best_weights},{"raw_logit_weights",raw},
                {"eval_weights",eval_weights},{"fixed_zero_features",{"jeopardy","tslot","belief_weights"}},
                {"max_well_depth",reader.teacher.at("max_well_depth")}};
            std::ofstream(output/"policy.json") << artifact.dump(2) << '\n';
            std::ofstream(output/"metrics.json") << report.dump(2) << '\n';
            std::cout << Json{{"epoch",epoch},{"validation",result},
                {"elapsed_seconds",report["elapsed_seconds"]}}.dump() << std::endl;
        };
        auto evaluate = [&] {
            Metrics metrics;
            for (const auto& file : validation) reader.read(file,[&](Example ex) { metrics.add(ex,model.w); });
            return metrics;
        };
        save(0,evaluate());
        for (int epoch=1; epoch<=epochs; ++epoch) {
            std::shuffle(train.begin(),train.end(),rng);
            const double rate = .01/std::sqrt(double(epoch));
            std::vector<Example> buffer;
            auto flush = [&] {
                std::shuffle(buffer.begin(),buffer.end(),rng);
                for (const auto& ex : buffer) model.add(ex,rate);
                buffer.clear();
            };
            for (size_t i=0; i<train.size(); ++i) {
                reader.read(train[i],[&](Example ex) {
                    buffer.push_back(std::move(ex));
                    if (buffer.size()==2048) flush();
                });
                if ((i+1)%50==0) std::cout << Json{{"epoch",epoch},{"train_games_read",i+1}}.dump() << std::endl;
            }
            flush(); model.flush(rate);
            save(epoch,evaluate());
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
