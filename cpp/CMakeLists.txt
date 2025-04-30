# CMakeList.txt : CMake project for cfr_tetris, include source and define
# project specific logic here.
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Tetris)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Tetris/game)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Tetris/bot)

add_subdirectory("Tetris/game/ShakTrisLib")


set(TETRIS_SOURCES
	"Tetris/bot/Eval.cpp"
	"Tetris/bot/Eval.hpp"
	"Tetris/game/EmulationGame.cpp"
	"Tetris/game/EmulationGame.hpp"
	"Tetris/game/Opponent.cpp"
	"Tetris/bot/UCT.cpp"
	"Tetris/bot/UCT.hpp"
	"Tetris/bot/Search.cpp"
	"Tetris/bot/Search.hpp"

	"Util/Distribution.cpp"
	"Util/fasthash.cpp"

	"icon.rc"
)
 

if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(PNG REQUIRED)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# Add source to this project's executable.
add_executable (cfr_tetris "cfr_tetris.cpp" "cfr_tetris.h" ${TETRIS_SOURCES})
add_executable(nana_bench "bench.cpp" ${TETRIS_SOURCES})
add_executable (tbp_tetris "tbp.cpp" ${TETRIS_SOURCES})

target_compile_definitions(tbp_tetris PRIVATE TBP TBP2)

if(UNIX AND NOT APPLE)
    target_include_directories(cfr_tetris PRIVATE ${X11_INCLUDE_DIR} ${OPENGL_INCLUDE_DIRS} ${PNG_INCLUDE_DIR})

    target_link_libraries(cfr_tetris ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${PNG_LIBRARY})
endif()

target_link_libraries(cfr_tetris ShakTris)
target_link_libraries(nana_bench ShakTris)
target_link_libraries(tbp_tetris ShakTris)

if (CMAKE_VERSION VERSION_GREATER 3.22)
  set_property(TARGET cfr_tetris PROPERTY CXX_STANDARD 23)
  set_property(TARGET nana_bench PROPERTY CXX_STANDARD 23)
  set_property(TARGET tbp_tetris PROPERTY CXX_STANDARD 23)
endif()

# TODO: Add tests and install targets if needed.
