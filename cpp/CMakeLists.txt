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

	"Util/Affinity.cpp"
	"Util/Allocator.cpp"
	"Util/Distribution.cpp"
	"Util/fasthash.cpp"

	"icon.rc"
)
 

# mimalloc for operator new/delete. glibc malloc grows per-thread heaps with mprotect,
# which serialises on the process' mmap lock: it cost ~25% at 32 threads on Linux
option(NANA_MIMALLOC "Use mimalloc for C++ allocations" ON)

if(NANA_MIMALLOC)
    include(FetchContent)
    set(MI_OVERRIDE OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(mimalloc
        GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
        GIT_TAG v3.5.4
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(mimalloc)
    # Util/Allocator.cpp swaps in mimalloc's operator new/delete
    add_compile_definitions(NANA_MIMALLOC)
    link_libraries(mimalloc-static)
endif()

# the graphical cfr_tetris needs X11, OpenGL and libpng on Linux;
# turn it off to build only the bots and benches on a headless machine
option(NANA_GUI "Build the graphical cfr_tetris" ON)

if(NANA_GUI AND UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(PNG REQUIRED)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# Add source to this project's executable.
if(NANA_GUI)
    add_executable (cfr_tetris "cfr_tetris.cpp" "cfr_tetris.h" ${TETRIS_SOURCES})
endif()
add_executable (tbp_tetris "tbp.cpp" ${TETRIS_SOURCES})
add_executable (nana_bench "bench.cpp" ${TETRIS_SOURCES})
target_compile_definitions(nana_bench PRIVATE NANA_BENCH)
add_executable (nana_msgbench "msgbench.cpp" ${TETRIS_SOURCES})

target_compile_definitions(tbp_tetris PRIVATE TBP TBP2)

if(NANA_GUI AND UNIX AND NOT APPLE)
    target_include_directories(cfr_tetris PRIVATE ${X11_INCLUDE_DIR} ${OPENGL_INCLUDE_DIRS} ${PNG_INCLUDE_DIR})

    target_link_libraries(cfr_tetris ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${PNG_LIBRARY})
endif()

if(NANA_GUI)
    target_link_libraries(cfr_tetris ShakTris)
endif()
target_link_libraries(tbp_tetris ShakTris)
target_link_libraries(nana_bench ShakTris)
target_link_libraries(nana_msgbench ShakTris)


if (CMAKE_VERSION VERSION_GREATER 3.12)
  if(NANA_GUI)
    set_property(TARGET cfr_tetris PROPERTY CXX_STANDARD 23)
  endif()
  set_property(TARGET tbp_tetris PROPERTY CXX_STANDARD 23)
  set_property(TARGET nana_bench PROPERTY CXX_STANDARD 23)
  set_property(TARGET nana_msgbench PROPERTY CXX_STANDARD 23)
endif()
