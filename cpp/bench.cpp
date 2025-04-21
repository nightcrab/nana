#include "EmulationGame.hpp"
#include "Search.hpp"


int main(int argc, const char** args) {
	// the arguments passed into the program will be in the format: <program> <core_count> <time in ms>
	if (argc < 3) {
		std::cerr << "Usage: " << args[0] << " <core_count> <time in ms>" << std::endl;
		return 1;
	}

	int core_count = std::stoi(args[1]);
	int time = std::stoi(args[2]);

	EmulationGame game;

	Search::startSearch(game, core_count);

	// run the search for the specified time
	std::this_thread::sleep_for(std::chrono::milliseconds(time));

	// end the search
	Search::endSearch();

	// check and print the stats
	Search::printStatistics();



	return 0;
}
