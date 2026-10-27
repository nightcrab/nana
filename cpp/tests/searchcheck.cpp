#include "SearchFixture.hpp"
#include "Tetris/bot/Search.hpp"
#include "Tetris/bot/Config.hpp"
#include "Util/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

using Json=nlohmann::json;
using namespace search_fixture;
static_assert(!cc_search);

int terminal(Model& m,double payoff) {
    const int id=int(m.nodes.size());
    m.nodes.push_back(Node{{},payoff,0});
    return id;
}
int chain(Model& m,int next,int length) {
    for(int i=0;i<length;++i) {
        const int id=int(m.nodes.size());
        m.nodes.push_back(Node{{Action{{next,1}}},0,0});
        next=id;
    }
    return next;
}
Model fixture(int kind) {
    Model m; m.nodes.emplace_back();
    if(kind==0) {
        m.name="forced-fair-coin";
        const int lo=terminal(m,0), hi=terminal(m,1);
        m.nodes[0].actions={Action{{lo,0.5},{hi,0.5}}};
    } else if(kind==1 || kind==2) {
        m.name=kind==1?"forced-rare-deep-outcome":"choose-safe-over-rare-deep";
        const int lo=terminal(m,0), hi=chain(m,terminal(m,1),5);
        m.nodes[0].actions={Action{{lo,0.875},{hi,0.125}}};
        if(kind==2) {
            const int safe=terminal(m,0.375);
            m.nodes[0].actions.push_back(Action{{safe,1}});
        }
    } else if(kind==3) {
        m.name="equal-value-different-fragmentation";
        const int direct=terminal(m,0.6);
        Action fragmented;
        for(int i=0;i<17;++i) {
            const int id=chain(m,terminal(m,0.6),2);
            fragmented.push_back({id,1.0/17});
        }
        m.nodes[0].actions={Action{{direct,1}},fragmented};
    } else if(kind==4) {
        m.name="seventeen-improving-children";
        Action a;
        for(int i=0;i<17;++i) {
            const int id=chain(m,terminal(m,0.2+0.6*i/16),3);
            a.push_back({id,1.0/17});
        }
        m.nodes[0].actions={a};
    } else {
        m.name="choose-profitable-stochastic-action";
        const int lo=terminal(m,0), hi=chain(m,terminal(m,1),4);
        const int safe=terminal(m,0.6);
        m.nodes[0].actions={Action{{lo,0.25},{hi,0.75}},Action{{safe,1}}};
    }
    return m;
}

Json run(const Model& m,int budget,int threads,int seed) {
    model=&m;
    for(auto& action:proposals) for(auto& p:action) p.store(0);
    fast_seed_reset(uint64_t(seed)+1);
    EmulationGame state;
    state.fixture_active=true;
    state.game.current_piece=Piece(PieceType::T);
    state.rng.PPTRNG=uint32_t(seed);
    Search::root_opponent=nullptr;
    Search::root_filter=nullptr;
    Search::simulation_limit=budget;
    Search::startSearch(state,threads);
    const bool reached=Search::waitForSimulations(std::chrono::seconds(30));
    Search::endSearch();
    if(!reached) throw std::runtime_error("fixture search timed out");
    UCTNode& root=Search::uct.getNode(state.hash());
    Json actions=Json::array();
    double maximum=0;
    std::vector<double> true_values;
    for(size_t a=0;a<m.nodes[0].actions.size();++a) {
        double truth=0;
        for(const auto& o:m.nodes[0].actions[a]) truth+=o.probability*exact(o.node);
        true_values.push_back(truth); maximum=std::max(maximum,truth);
        Reward r;
        const double estimate=root.armValue(a,r);
        const auto& table=root.cold[a].children;
        float count=0; table.sum(count);
        Json unexpected=Json::array();
        auto check_child=[&](const ChildValue& child) {
            if(child.n<=0) return;
            const auto& expected=m.nodes[0].actions[a];
            const bool valid=std::any_of(expected.begin(),expected.end(),[&](const Outcome& o) {
                return child.hash==hash_base+o.node;
            });
            if(!valid) unexpected.push_back({{"hash",child.hash},{"node",child.hash-hash_base},
                {"weight",child.n},{"value",rewardScalar(child.V)}});
        };
        check_child(table.first);
        if(table.more) for(int k=0;k<ChildTable::MORE;++k) check_child(table.more[k]);
        Json outcomes=Json::array();
        uint64_t total_draws=0;
        for(size_t j=0;j<m.nodes[0].actions[a].size();++j) total_draws+=proposals[a][j].load();
        for(size_t j=0;j<m.nodes[0].actions[a].size();++j) {
            const auto& o=m.nodes[0].actions[a][j];
            const uint32_t h=hash_base+o.node;
            double weight=0; bool cached=false;
            if(table.first.n>0 && table.first.hash==h) { weight=table.first.n; cached=true; }
            if(table.more) for(int k=0;k<ChildTable::MORE;++k)
                if(table.more[k].n>0 && table.more[k].hash==h) { weight=table.more[k].n; cached=true; }
            outcomes.push_back({{"probability",o.probability},{"proposals",proposals[a][j].load()},
                {"proposal_frequency",total_draws?double(proposals[a][j].load())/total_draws:0},
                {"cached",cached},{"backup_weight",weight},{"backup_frequency",count>0?weight/count:0}});
        }
        actions.push_back({{"exact",truth},{"estimate",estimate},{"error",estimate-truth},
            {"visits",root.stats(a).N},{"proposal_count",total_draws},
            {"backup_count",count},{"overflow_weight",table.rest_n},{"outcomes",outcomes},
            {"unexpected_children",unexpected}});
    }
    const Move chosen=Search::bestMove();
    const int action=chosen.piece.position.x;
    uint64_t restarts=0,replay_matched=0,replay_diverged=0,flushes=0;
    for(const auto& s:Search::uct.stats) {
        restarts+=s.restarts; replay_matched+=s.replay_matched;
        replay_diverged+=s.replay_diverged; flushes+=s.flushes;
    }
    return {{"fixture",m.name},{"budget",budget},{"threads",threads},{"seed",seed},
        {"chosen",action},{"regret",maximum-true_values.at(action)},
        {"actions",actions},{"restarts",restarts},{"replay_matched",replay_matched},
        {"replay_diverged",replay_diverged},{"flushes",flushes}};
}

int main(int argc,char** argv) {
    try {
        const int repeats=argc>1?std::stoi(argv[1]):12;
        inject_weights=argc>2 && std::string(argv[2])=="--inject-weights";
        if(repeats<4) throw std::runtime_error("use at least four independent search seeds");
        Json all=Json::array(),groups=Json::array();
        int failures=0;
        for(int kind=0;kind<(inject_weights?1:6);++kind) {
            const auto m=fixture(kind); model=&m;
            for(int threads:{1,4}) for(int budget:{2048,32768}) {
                double wrong=0,regret=0;
                int invalid_children=0;
                std::vector<double> errors(m.nodes[0].actions.size(),0);
                std::vector<double> squares(errors.size(),0);
                for(int rep=0;rep<repeats;++rep) {
                    const auto row=run(m,budget,threads,7000+rep*7919);
                    const double loss=row["regret"];
                    regret+=loss; wrong+=loss>1e-6;
                    for(size_t a=0;a<errors.size();++a) {
                        const double e=row["actions"][a]["error"];
                        errors[a]+=e; squares[a]+=e*e;
                        invalid_children+=int(row["actions"][a]["unexpected_children"].size());
                    }
                    all.push_back(row);
                }
                bool ok=invalid_children==0;
                Json calibration=Json::array();
                for(size_t a=0;a<errors.size();++a) {
                    const double bias=errors[a]/repeats;
                    const double se=std::sqrt(std::max(0.0,(squares[a]-errors[a]*errors[a]/repeats)/(repeats-1))/repeats);
                    calibration.push_back({{"action",a},{"mean_error",bias},{"standard_error",se}});
                    if(budget==32768 && std::abs(bias)>0.04) ok=false;
                }
                if(budget==32768 && wrong/repeats>0.1) ok=false;
                failures+=!ok;
                groups.push_back({{"fixture",m.name},{"threads",threads},{"budget",budget},
                    {"status",!ok?"fail":budget==2048?"screen":"pass"},
                    {"unexpected_children",invalid_children},
                    {"wrong_choice_fraction",wrong/repeats},{"mean_regret",regret/repeats},
                    {"calibration",calibration}});
                std::cerr<<m.name<<" threads="<<threads<<" budget="<<budget<<" "<<(ok?"ok":"FAIL")<<'\n';
            }
        }
        std::cout<<Json{{"repeats",repeats},{"injected_weights",inject_weights},
            {"failures",failures},{"groups",groups},{"runs",all}}.dump(2)<<'\n';
        return failures?1:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}
