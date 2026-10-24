#pragma once

#include "engine/Board.hpp"
#include "engine/Game.hpp"
#include "Move.hpp"

#include <vector>

namespace Eval {
	double eval_CC(const Board& board, int lines, bool tspin, bool waste_t);
	// move_part returns the share rewarding the move itself, on the evaluator's scale.
	double eval_CC(Game game, Move move, double* move_part = nullptr);
	// Cold Clear Standard evaluator; bag_left is the bag remaining after the preview queue.
	double eval_CC_standard(Game game, Move move, const std::vector<PieceType>& bag_left,
	                        double* move_part = nullptr);

	[[maybe_unused]] static bool is_top_quarter(const Board& board);

	[[maybe_unused]] static bool is_top_half(const Board& board);

	std::pair<int, int> cavities_overhangs(const Board& board);

	int well_position(const Board& board);

	// lowest height, highest height
	std::pair<int, int> height_features(const Board& board);

	std::pair<int, int> n_covered_cells(Board board);

	int get_row_transitions(const Board& board);

	[[maybe_unused]] static std::pair<int, int> get_bumpiness(const Board& board);
	// Identify clean count to 4

	[[maybe_unused]] static bool ct4(const Board& board);

};  // namespace Eval