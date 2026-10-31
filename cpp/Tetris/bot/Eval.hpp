#pragma once

#include "Config.hpp"
#include "engine/Board.hpp"
#include "engine/Game.hpp"
#include "Move.hpp"
#include "Belief.hpp"

#include <cstdint>
#include <vector>

// Every eval scores placing a piece: 1 is neutral, 0 is death. move_part, when given, receives the share that
// belongs to the move itself (clears, spins, attack) rather than the board left behind.
namespace Eval {
	struct BeliefTerm {
		const opponent::Belief* filter = nullptr;
		int meter = 0;
		std::array<float, 32> cache;
		BeliefTerm(const opponent::Belief* filter, int meter);
		float value(int attack, bool cleared);
	};

	// what the evals of one node's candidate moves share, built once per node
	struct Context {
		const Game& game;
		EvalMode mode;
		std::vector<PieceType> bag_left;  // the bag past the preview
		BeliefTerm* belief = nullptr;     // opponent belief term
		int meter = 0;                    // incoming garbage
	};

	struct BoardFeatures {
		int bottom, height, cavities, overhangs, covered, covered_sq, bumpiness, bumpiness_sq;
		int transitions, well, well_depth;
		int high_cavities, spawn_height;
		bool top_half, top_quarter, low, perfect_clear, quad_ready;
	};
	struct MoveFeatures {
		int lines, attack, time;
		spinType spin;
		bool perfect_clear, waste_t, back_to_back, b2b_clear;
	};
	BoardFeatures boardFeatures(const Board& board);

	// Opponent conditioning for the current search (all zero = off). Only written while no search runs.
	struct ConditioningDirection {
		// w = Σ_k g_k v_k, indexed by PlacementFeature
		std::array<double, PlacementFeature::COUNT> placement{};
		// Σ_k g_k opp_cond_reward[k], added to app_weight and eval_weight
		double app = 0, eval = 0;
		bool operator==(const ConditioningDirection&) const = default;
	};
	// The tree carries over between moves, so values backed up under earlier directions stay mixed in.
	inline ConditioningDirection opponent_direction{};
	// Each term's gate g_k from the opponent's visible state, one per opp_cond row
	std::vector<float> opponentGates(const Board& board, int meter, int b2b, int combo);
	// the direction from the opponent's visible state; zero unless some row has a nonzero u and a nonzero v or
	// reward shift
	ConditioningDirection opponentDirection(const Board& board, int meter, int b2b, int combo);
	void setOpponentDirection(const ConditioningDirection& direction);
	// `after` already has the piece placed and lines cleared; damage_sent updates its combo/B2B state.
	MoveFeatures moveFeatures(const Game& parent, const Piece& piece, Game& after, int lines);

	struct ActionEvaluation {
		float value, move_reward;
		bool convex;
	};
	// One placement and feature pass for the leaf value and move reward caches.
	ActionEvaluation evaluateAction(const Context& context, const Piece& piece);

	// the eval of context.mode; convex, when given, receives whether the board left behind is surface convex
	// (always true outside the convex eval)
	double evaluate(const Context& context, const Piece& piece, double* move_part = nullptr, bool* convex = nullptr);

	// nana's eval
	double evalNana(const Game& game, const Piece& piece, double* move_part = nullptr);
	double evalNana(const Context& context, const Piece& piece, double* move_part = nullptr,
	                bool* convex = nullptr);
	// nana's eval of a board after placing a piece and clearing lines
	double nanaBoard(const Board& board, int lines, bool tspin, bool waste_t);
	// nana's eval for a no-spin style: one low score for every board that isn't surface convex, nana's eval without
	// the spin features for the rest, never below a live board's floor just above death
	double evalConvex(const Game& game, const Piece& piece, double* move_part = nullptr, bool* convex = nullptr);

	[[maybe_unused]] static bool is_top_quarter(const Board& board);

	[[maybe_unused]] static bool is_top_half(const Board& board);

	std::pair<int, int> cavities_overhangs(const Board& board);

	int well_position(const Board& board);

	// lowest height, highest height
	std::pair<int, int> height_features(const Board& board);

	std::pair<int, int> n_covered_cells(Board board);

	int get_row_transitions(const Board& board);
	int well_depth(const Board& board, int well, int cap);

	[[maybe_unused]] static std::pair<int, int> get_bumpiness(const Board& board);
	// Identify clean count to 4

	bool ct4(const Board& board);

};  // namespace Eval
