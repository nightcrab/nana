#include "EmulationGame.hpp"
#include "engine/Board.hpp"
#include "engine/ShaktrisConstants.hpp"
#include "Search.hpp"

#include <array>
#include <chrono>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

struct shim_game {
	Board board;
	std::array<PieceType, 6> queue;
	std::optional<PieceType> hold;
	PieceType current_piece;
};

constexpr std::array<shim_game, 2> test_games = []() {
	Board first_board;
	first_board.board = {
		0b00011111,
		0b00011111,
		0b00111101,
		0b00011011,
		0b00111111,
		0b00011110,
		0b00011111,
		0b00000111,
		0b01111111,
		0b11111111,
	};

	Board second_board;
	second_board.board = {
		0b1111111111,
		0b1111111111,
		0b1111011111,
		0b1011111111,
		0b0011111111,
		0b0011100000,
		0b0000111111,
		0b0111111111,
		0b0111111111,
		0b0111111111,
	};

	std::array<shim_game, 2> games{{
		{
			first_board,
			{
				PieceType::Z,
				PieceType::L,
				PieceType::S,
				PieceType::S,
				PieceType::O,
				PieceType::Empty
			},
			PieceType::S,
			PieceType::O,
		},
		{
			second_board,
			{
				PieceType::L,
				PieceType::Z,
				PieceType::O,
				PieceType::S,
				PieceType::I,
				PieceType::Empty
			},
			PieceType::O,
			PieceType::J,
		}
	}};

	return games;
}();

// print biggest N and R
// code stolen from the print statistics function
void print_strength() {
	int biggest_N = 0;
	float biggest_R = 0;
	Move best_move;

	for (Action& action : Search::uct.getNode(Search::root_state.hash()).actions) {
		if constexpr (Search::search_style == NANA) {
			if (action.N == biggest_N) {
				if (action.R > biggest_R) {
					biggest_R = action.R;
					biggest_N = action.N;
					best_move = action.move;
				}
			}
			if (action.N > biggest_N) {
				biggest_R = action.R;
				biggest_N = action.N;
				best_move = action.move;
			}
		}
		if constexpr (Search::search_style == CC) {
			if (action.R > biggest_R) {
				biggest_R = action.R;
				biggest_N = action.N;
				best_move = action.move;
			}
		}


		}
	
	std::cout << biggest_R << std::endl;
}

int main(int argc, const char** args) {
	// for debugging in vs22
	if (false) {
		const char* argss[] = {"executable", "1","10000","0"};
		args = argss;
		argc = sizeof(argss) / sizeof(*argss);
	}
	// the arguments passed into the program will be in the format: <program> <core_count> <time in ms> <test_number> 0 or 1
	if (argc < 4) {
		std::cerr << "Usage: " << args[0] << " <core_count> <time in ms> <test_number [0,1]>" << std::endl;
		return 1;
	}
	int core_count 	= 0;
	int time 		= 0;
	int test_number = 0;
	
	try {
		core_count = std::stoi(args[1]);
		time = std::stoi(args[2]);
		test_number = std::stoi(args[3]);
		shim_game tmp = test_games.at(test_number);

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	EmulationGame game;

	game.game.board = test_games.at(test_number).board;
	game.game.queue = test_games.at(test_number).queue;
	game.game.hold = test_games.at(test_number).hold;
	game.game.current_piece = test_games.at(test_number).current_piece;

	Search::startSearch(game, core_count);

	// run the search for the specified time
	std::this_thread::sleep_for(std::chrono::milliseconds(time));

	// end the search
	Search::endSearch();

	// check and print the stats
	print_strength();
	Search::printStatistics();

	return 0;
}
