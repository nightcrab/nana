#pragma once

#include "Config.hpp"
#include "engine/Board.hpp"
#include "engine/Game.hpp"
#include "Move.hpp"
#include "OpponentFilter.hpp"

#include <vector>

// Every eval scores placing a piece: 1 is neutral, 0 is death. move_part, when given, receives the share that
// belongs to the move itself (clears, spins, attack) rather than the board left behind.
namespace Eval {
	struct BeliefTerm {
		const OpponentFilter* filter = nullptr;
		int meter = 0;
		std::array<float, 32> cache;
		BeliefTerm(const OpponentFilter* filter, int meter);
		float value(int attack, bool cleared);
	};

	// what the evals of one node's candidate moves share, built once per node
	struct Context {
		const Game& game;
		EvalMode mode;
		std::vector<PieceType> bag_left;  // cc: the bag past the preview
		BeliefTerm* belief = nullptr;     // cc: opponent belief term
		int meter = 0;                    // cc: incoming garbage
	};

	// the eval of context.mode; convex, when given, receives whether the board left behind is surface convex
	// (always true outside the convex eval)
	double evaluate(const Context& context, const Piece& piece, double* move_part = nullptr, bool* convex = nullptr);

	// Cold Clear's eval
	double evalCC(const Game& game, const Piece& piece, const std::vector<PieceType>& bag_left,
	              double* move_part = nullptr, BeliefTerm* belief = nullptr, int meter = 0);
	// nana's eval
	double evalNana(const Game& game, const Piece& piece, double* move_part = nullptr);
	// nana's eval of a board after placing a piece and clearing lines
	double nanaBoard(const Board& board, int lines, bool tspin, bool waste_t);
	// nana's eval for a no-spin style: one low score for every board that isn't surface convex, nana's eval without
	// the spin features for the rest, never below a live board's floor just above death
	double evalConvex(const Game& game, const Piece& piece, double* move_part = nullptr, bool* convex = nullptr);

	// evalCC split into its weighted terms, in cc points; they sum to (total - 1) * 10000
	inline constexpr const char* ccTermNames[] = {
		"perfect_clear", "b2b_clear", "combo_garbage", "clear", "spin_clear", "wasted_t", "move_time", "jeopardy",
		"back_to_back", "top_quarter", "top_half", "tslot", "height", "well_depth", "well_column", "row_transitions",
		"bumpiness", "bumpiness_sq", "cavity_cells", "cavity_cells_sq", "overhang_cells", "overhang_cells_sq",
		"covered_cells", "covered_cells_sq",
	};
	inline constexpr int CC_TERMS = (int)std::size(ccTermNames);
	double evalCCTerms(const Game& game, const Piece& piece, const std::vector<PieceType>& bag_left,
	                   std::array<double, CC_TERMS>& terms);
	// evalCC compiled with only some terms, for timing feature subsets: a few named subsets, then each term alone on
	// top of the work every subset shares
	int ccTimedCount();
	// the named subsets come first; the last of them computes no terms
	int ccTimedSets();
	const char* ccTimedName(int i);
	double evalCCTimed(int i, const Game& game, const Piece& piece, const std::vector<PieceType>& bag_left);

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