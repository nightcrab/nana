#include "Tetris/bot/Eval.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

static void check(bool ok, const char* message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
static bool near(double a, double b) { return std::abs(a - b) < 1e-9; }

static void boardChecks() {
    Board b;
    check(Eval::get_row_transitions(b) == 64, "empty board: two walls over 32 rows");
    check(!Eval::ct4(b), "empty board is not quad ready");
    b.board[4] = 1;
    check(Eval::get_row_transitions(b) == 66, "both interior boundaries must be counted");
    b.board.fill(0);
    b.board[4] = 0b10101;
    check(Eval::n_covered_cells(b) == std::pair{3, 5}, "count occupied cells above every hole");
    b.board.fill(15); b.board[9] = 0;
    check(Eval::ct4(b), "four rows beside an empty well");
    check(Eval::well_depth(b, 9, 17) == 4, "four-row well depth");
    check(Eval::well_depth(b, 9, 2) == 2, "well depth cap");
    b.board[0] &= ~2u;
    check(!Eval::ct4(b), "buried gap prevents a clean quad");
    b.board.fill(63); b.board[0] = 3;
    check(Eval::ct4(b), "quad above a common base");
    b.board.fill(UINT32_MAX);
    check(Eval::cavities_overhangs(b) == std::pair{0, 0}, "full-height solid columns");
    check(Eval::n_covered_cells(b) == std::pair{0, 0}, "full-height columns cover no holes");
    check(Eval::well_depth(b, 0, 17) == 0, "well above visible field");
    std::mt19937 rng(7129);
    for (int trial = 0; trial < 2000; ++trial) {
        for (auto& col : b.board) col = trial % 2 ? rng() : rng() & 0xffff;
        int heights[10]{};
        for (int x = 0; x < 10; ++x) {
            for (int y = 0; y < 32; ++y) if ((b.board[x] >> y) & 1) heights[x] = y + 1;
        }
        int holes = 0, overhangs = 0, covered = 0, squares = 0, transitions = 0;
        for (int y = 0; y < 32; ++y) {
            bool previous = true;
            for (int x = 0; x < 10; ++x) {
                bool filled = (b.board[x] >> y) & 1;
                transitions += previous != filled;
                previous = filled;
                if (!filled && y < heights[x]) {
                    ++holes;
                    overhangs += (x > 0 && heights[x - 1] <= y) || (x < 9 && heights[x + 1] <= y);
                    int above = 0;
                    for (int z = y + 1; z < 32; ++z) above += (b.board[x] >> z) & 1;
                    covered += above; squares += above * above;
                }
            }
            transitions += !previous;
        }
        check(Eval::cavities_overhangs(b) == std::pair{holes, overhangs}, "hole/overhang scalar reference");
        check(Eval::n_covered_cells(b) == std::pair{covered, squares}, "covered-cell scalar reference");
        check(Eval::get_row_transitions(b) == transitions, "transition scalar reference");
        Board mirror;
        for (int x = 0; x < 10; ++x) mirror.board[x] = b.board[9 - x];
        check(Eval::cavities_overhangs(b) == Eval::cavities_overhangs(mirror), "overhang reflection symmetry");
    }
}

static void moveChecks() {
    const auto& c = config();
    check(c.b2b_clear != 0 && c.combo_garbage != 0 && c.move_time != 0 && c.mini_tspins[1] != 0,
          "run with NANA_CONFIG=cpp/tests/evalcheck.json to exercise enabled features");
    Game g;
    g.mode = TetrioS1();
    g.current_piece = PieceType::I;
    for (int x = 0; x < 6; ++x) g.board.board[x] = 1;
    Piece pc(PieceType::I);
    pc.position = {7, 0};
    pc.minos = {Coord{-1, 0}, Coord{0, 0}, Coord{1, 0}, Coord{2, 0}};
    double move = 0;
    const double total = Eval::evalNana(g, pc, &move);
    Game after = g;
    after.place_piece(pc);
    const int lines = after.board.clearLines();
    const int attack = after.damage_sent(lines, spinType::null, true);
    check(after.board.is_empty(), "fixture makes a perfect clear");
    check(near(move * 10000, c.clears[1] + c.perfect_clear + c.combo_garbage * attack + c.move_time * 43),
          "PC and attack belong to move reward");
    check(near(total - move, Eval::nanaBoard(Board{}, 0, false, false) - c.perfect_clear / 10000),
          "PC bonus does not remain in board value");
    check(g.board.board[0] == 1 && g.combo == 0, "evaluation must not mutate the game");

    g.board.board.fill(15); g.board.board[9] = 0; g.board.board[0] |= 16;
    g.b2b = 1; g.combo = 2;
    Piece quad(PieceType::I, RotationDirection::East, Coord{9, 1});
    quad.minos = {Coord{0, -1}, Coord{0, 0}, Coord{0, 1}, Coord{0, 2}};
    after = g; after.place_piece(quad);
    after.board.clearLines();
    const int quad_attack = after.damage_sent(4, spinType::null, false);
    const double quad_total = Eval::evalNana(g, quad, &move);
    check(near(move * 10000, c.clears[4] + c.b2b_clear + c.combo_garbage * quad_attack + c.move_time * 46),
          "B2B quad, combo attack and movement time");
    check(near(quad_total - move, Eval::nanaBoard(after.board, 0, false, false) + c.back_to_back / 10000),
          "maintained B2B is position value");

    g = Game{}; g.mode = TetrioS1(); g.current_piece = PieceType::T;
    g.board.board.fill(1); g.board.board[3] = g.board.board[4] = g.board.board[5] = 0;
    Piece spin(PieceType::T, RotationDirection::North, Coord{4, 0}, spinType::normal);
    double full_move = 0;
    Eval::evalNana(g, spin, &full_move);
    spin.spin = spinType::mini;
    Eval::evalNana(g, spin, &move);
    Game full_after = g; full_after.place_piece(spin); full_after.board.clearLines();
    Game mini_after = full_after;
    const int full_attack = full_after.damage_sent(1, spinType::normal, false);
    const int mini_attack = mini_after.damage_sent(1, spinType::mini, false);
    check(near((full_move - move) * 10000, c.tspins[1] - c.mini_tspins[1] - c.wasted_t
               + c.combo_garbage * (full_attack - mini_attack)), "full and mini spin bonuses stay additive");
    bool convex = true;
    g.board.board.fill(0); g.board.board[0] = 4;
    spin.position = {4, 0};
    const double rejected = Eval::evalConvex(g, spin, &move, &convex);
    check(!convex && move == 0 && near(rejected, std::max(1 + c.nonconvex / 10000, 0.0001)),
          "convex style still rejects nonconvex boards");

    g = Game{}; g.mode = TetrioS1(); g.current_piece = PieceType::O; g.hold = PieceType::T;
    g.board.board[0] = 0b1010; g.board.board[2] = 0b0011;
    Piece o(PieceType::O, RotationDirection::North, Coord{7, 1});
    after = g; after.place_piece(o); after.board.clearLines();
    const double slot_total = Eval::evalNana(g, o, &move);
    check(near(slot_total - move, Eval::nanaBoard(after.board, 0, false, false) + c.tslot[0] / 10000),
          "slot potential is separate from actual board features");

    g.board.board.fill(0); g.board.board[0] = 1u << 12;
    const double high_total = Eval::evalNana(g, o, &move);
    check(near(move * 10000, c.move_time * 3 + std::trunc(c.jeopardy * 3 * 3 / 10)),
          "jeopardy couples move time with height above ten");
    OpponentFilter filter;
    Eval::BeliefTerm belief(&filter, 6);
    Eval::Context context{g, EvalMode::Nana}; context.belief = &belief; context.meter = 6;
    double belief_move = 0;
    check(near(Eval::evaluate(context, o, &belief_move) - high_total, belief.value(0, false) / 10000),
          "opponent belief contributes position value");
    check(near(move, belief_move), "belief must not enter move reward");
    belief.value(15, true);
    check(near(belief.value(20, true), filter.valueAfter(20, true, 0, c.belief_weights)),
          "large attacks do not alias the attack-15 belief cache");
}

int main() {
    boardChecks();
    moveChecks();
    std::cout << "Evaluator checks passed (2000 scalar-reference boards and scoring fixtures).\n";
}
