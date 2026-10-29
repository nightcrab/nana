// Dumps node priors and cached leaf scores for independent rank checks.
#include "EmulateCommon.hpp"
#include "UCT.hpp"
#include "Util/json.hpp"
#include <iostream>

int main() {
    for (int sample = 0; sample < 3; ++sample) {
        EmulationGame game = newGame(100 + sample, 0, 1);
        if (sample > 0) {
            game.game.board.board.fill(0b1111);
            game.game.board.board[4] = 0;
            game.game.current_piece = Piece(PieceType::I);
            game.game.hold = PieceType::T;
            game.game.b2b = 2;
            game.game.combo = 3;
            if (sample == 2) game.game.add_garbage(3, 7);
        }
        UCTNode node(game);
        nlohmann::json actions = nlohmann::json::array();
        for (size_t i = 0; i < node.actionCount(); ++i) {
            actions.push_back({{"prior", node.stats(i).prior},
                {"eval", node.eval(i)}, {"move_reward", node.cold[i].move_reward}});
        }
        std::cout << actions.dump() << '\n';
    }
}
