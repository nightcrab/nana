#include "OpponentFeatures.hpp"

#include "EmulationGame.hpp"
#include "Eval.hpp"
#include "engine/Utility.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace opponent {

const char* featureName(int feature) {
    static const char* const names[FEATURES] = {
        "height", "well_depth", "row_transitions", "bumpiness", "cavities", "overhangs", "covered", "tslot",
        "tslot_lines", "combo", "b2b", "meter", "best_attack_now",
        "rollout_attack_0", "rollout_attack_1", "rollout_attack_2", "rollout_attack_3", "rollout_attack_4",
        "rollout_attack_5", "rollout_dead",
    };
    static_assert(ROLLOUT == 6, "featureName lists six rollout placements");
    return feature >= 0 && feature < FEATURES ? names[feature] : "?";
}

namespace {

int attackOf(const Game& before, const Piece& piece) {
    Game game = before;
    game.place_piece(piece);
    const int lines = game.board.clearLines();
    spinType spin = piece.spin;
    if (std::holds_alternative<TetrioS1>(game.mode) && piece.type != PieceType::T) {
        spin = spinType::null;
    }
    bool pc = true;
    for (int x = 0; x < Board::width; x++) {
        pc = pc && game.board.get_column(x) == 0;
    }
    return lines > 0 || pc ? game.damage_sent(lines, spin, pc) : 0;
}

}

std::array<float, FEATURES> features(const View& view) {
    std::array<float, FEATURES> out{};

    const Eval::CCBoardTerms terms = Eval::cc_board_terms(view.board);
    out[HEIGHT] = (float)terms.height;
    out[WELL_DEPTH] = (float)terms.well_depth;
    out[ROW_TRANSITIONS] = (float)terms.row_transitions;
    out[BUMPINESS] = (float)terms.bumpiness;
    out[CAVITIES] = (float)terms.cavities;
    out[OVERHANGS] = (float)terms.overhangs;
    out[COVERED] = (float)terms.covered;
    out[TSLOT] = terms.tslot ? 1.0f : 0.0f;
    out[TSLOT_LINES] = (float)terms.tslot_lines;
    out[COMBO] = (float)view.combo;
    out[B2B] = (float)view.b2b;
    out[METER] = (float)view.meter;

    // The rollout plays the known queue with Cold Clear's default weights and no incoming garbage (Cold Clear's
    // eval ignores the meter). The queue slot past the previews is never placed, so its fill doesn't matter.
    EmulationGame game(false);
    game.set_tetrio_rules();
    game.garbage_delay = 0;
    game.rng.PPTRNG = 0x9E3779B9u;
    game.rng.makebag();
    game.game.board = view.board;
    game.game.current_piece = Piece(view.current);
    game.game.hold = view.hold;
    game.game.queue.fill(PieceType::Empty);
    std::copy(view.queue.begin(), view.queue.end(), game.game.queue.begin());
    game.game.b2b = (u16)std::max(view.b2b, 0);
    game.game.combo = (u16)std::max(view.combo, 0);

    static const CCWeights cold_clear{};
    const std::vector<PieceType> bag_left;
    for (int step = 0; step < ROLLOUT; step++) {
        if (game.game.current_piece.type == PieceType::Empty ||
            Shaktris::Utility::collides(game.game.board, game.game.current_piece)) {
            out[ROLLOUT_DEAD] = 1;
            break;
        }
        const std::vector<Piece> placements = game.placements();
        if (placements.empty()) {
            out[ROLLOUT_DEAD] = 1;
            break;
        }
        double best = -std::numeric_limits<double>::infinity();
        size_t best_index = 0;
        int best_attack = 0;
        for (size_t i = 0; i < placements.size(); i++) {
            const double value = Eval::eval_CC_standard(cold_clear, game.game, Move(placements[i], false), bag_left);
            if (value > best) {
                best = value;
                best_index = i;
            }
            if (step == 0) {
                best_attack = std::max(best_attack, attackOf(game.game, placements[i]));
            }
        }
        if (step == 0) {
            out[BEST_ATTACK_NOW] = (float)best_attack;
        }
        const int attack_before = game.attack;
        game.set_move(Move(placements[best_index], false));
        game.play_moves<NoOpponent>();
        out[ROLLOUT_ATTACK + step] = (float)(game.attack - attack_before);
        if (game.game_over) {
            out[ROLLOUT_DEAD] = 1;
            break;
        }
        game.chance_move<NoOpponent>();
    }
    return out;
}

}
