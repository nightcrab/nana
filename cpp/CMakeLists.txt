# CMakeList.txt : CMake project for nana, include source and define
# project specific logic here.
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Tetris)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Tetris/game)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Tetris/bot)

add_subdirectory("Tetris/game/ShakTrisLib" EXCLUDE_FROM_ALL)
get_target_property(SHAKTRIS_DIR ShakTris SOURCE_DIR)
get_target_property(SHAKTRIS_SOURCES ShakTris SOURCES)
list(TRANSFORM SHAKTRIS_SOURCES PREPEND "${SHAKTRIS_DIR}/")


set(TETRIS_SOURCES
	"Tetris/bot/Eval.cpp"
	"Tetris/bot/Eval.hpp"
	"Tetris/bot/Config.cpp"
	"Tetris/bot/Config.hpp"
	"Tetris/game/EmulationGame.cpp"
	"Tetris/game/EmulationGame.hpp"
	"Tetris/game/MarkovOpponent.cpp"
	"Tetris/game/OpponentEncoder.cpp"
	"Tetris/bot/UCT.cpp"
	"Tetris/bot/UCT.hpp"
	"Tetris/bot/Search.cpp"
	"Tetris/bot/Search.hpp"

	"Util/Affinity.cpp"
	"Util/Allocator.cpp"
	"Util/Distribution.cpp"
	"Util/fasthash.cpp"
)
set(NANA_ICON "icon.rc")


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
    add_compile_definitions(NANA_MIMALLOC)
    link_libraries(mimalloc-static)
endif()

option(NANA_GUI "Build the graphical nana_interactive" ON)

if(NANA_GUI AND UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(PNG REQUIRED)
endif()


set(NANA_MARCH "native" CACHE STRING "-march for the benches, emulator and interactive build")
set(NANA_VARIANTS "" CACHE STRING "ISA levels to build nana_tbp for, behind a launcher")
set(NANA_KNOWN_VARIANTS znver5 znver4 x86-64-v4 x86-64-v3 x86-64-v2 x86-64)

function(nana_arch_flags out march)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(march MATCHES "^(x86-64-v4|znver[45])$")
            set(${out} "/arch:AVX512" PARENT_SCOPE)
        elseif(march MATCHES "^(native|x86-64-v3)$")
            set(${out} "/arch:AVX2" PARENT_SCOPE)
        elseif(march STREQUAL "x86-64-v2")
            set(${out} "/arch:SSE4.2" PARENT_SCOPE)
        else()
            set(${out} "" PARENT_SCOPE)
        endif()
    else()
        set(${out} "-march=${march}" PARENT_SCOPE)
    endif()
endfunction()

function(nana_add_core name march)
    add_library(${name} OBJECT ${TETRIS_SOURCES} ${SHAKTRIS_SOURCES})
    nana_arch_flags(flags ${march})
    target_compile_options(${name} PUBLIC ${flags})
    target_include_directories(${name} PUBLIC ${SHAKTRIS_DIR})
    set_property(TARGET ${name} PROPERTY CXX_STANDARD 23)
endfunction()

function(nana_add_executable name core)
    add_executable(${name} ${ARGN} ${NANA_ICON})
    target_link_libraries(${name} ${core})
    set_property(TARGET ${name} PROPERTY CXX_STANDARD 23)
endfunction()

nana_add_core(nana_core ${NANA_MARCH})
nana_add_core(nana_core_bench ${NANA_MARCH})
target_compile_definitions(nana_core_bench PUBLIC NANA_BENCH)

if(NANA_GUI)
    nana_add_executable(nana_interactive nana_core "interactive.cpp")
endif()
nana_add_executable(nana_bench nana_core_bench "bench.cpp")
nana_add_executable(nana_speed nana_core "bench.cpp")
nana_add_executable(nana_msgbench nana_core "msgbench.cpp")
nana_add_executable(nana_emulate nana_core "emulate.cpp")
nana_add_executable(nana_emulate_tbp nana_core "emulate_tbp.cpp")
nana_add_executable(nana_eval_dump nana_core "eval_dump.cpp")

option(NANA_EVALCHECK "Build evaluator regression checks" OFF)
if(NANA_EVALCHECK)
    nana_add_executable(nana_evalcheck nana_core "evalcheck.cpp")
endif()

option(NANA_STATCHECK "Build exact-reference sampling and backup diagnostics" OFF)
if(NANA_STATCHECK)
    nana_add_executable(nana_statcheck nana_core "statcheck.cpp")
endif()

option(NANA_SEARCHCHECK "Build full scheduler exact-value integration tests" OFF)
if(NANA_SEARCHCHECK)
    nana_add_core(nana_core_searchcheck ${NANA_MARCH})
    target_compile_definitions(nana_core_searchcheck PUBLIC NANA_SEARCH_FIXTURE=1)
    target_sources(nana_core_searchcheck PRIVATE "SearchFixture.cpp")
    nana_add_executable(nana_searchcheck nana_core_searchcheck "searchcheck.cpp")
endif()

# nana_cctree: the experimental CC search with the unified nana evaluator.
# NANA_CCTREE_LOAD_FACTOR 1 runs one simulation at a time.
option(NANA_CCTREE "Build nana_cctree, CC-search tree dumps using the unified evaluator" OFF)
set(NANA_CCTREE_LOAD_FACTOR 1 CACHE STRING "jobs in flight in nana_cctree's search")
if(NANA_CCTREE)
    nana_add_core(nana_core_cctree ${NANA_MARCH})
    target_compile_definitions(nana_core_cctree PUBLIC NANA_CC_SEARCH=1 NANA_CC_LOCKSTEP=1
        NANA_LOAD_FACTOR=${NANA_CCTREE_LOAD_FACTOR})
    nana_add_executable(nana_cctree nana_core_cctree "cctree.cpp")
endif()

if(NANA_VARIANTS)
    foreach(variant IN LISTS NANA_VARIANTS)
        if(NOT variant IN_LIST NANA_KNOWN_VARIANTS)
            message(FATAL_ERROR "NANA_VARIANTS: the launcher can't detect ${variant}; use some of ${NANA_KNOWN_VARIANTS}")
        endif()
        nana_add_core(nana_core_${variant} ${variant})
        nana_add_executable(nana_tbp-${variant} nana_core_${variant} "tbp.cpp")
        target_compile_definitions(nana_tbp-${variant} PRIVATE TBP TBP2)
    endforeach()

    add_executable(nana_tbp "launcher.cpp" "Tetris/bot/Config.cpp" ${NANA_ICON})
    list(JOIN NANA_VARIANTS "," variants)
    target_compile_definitions(nana_tbp PRIVATE "NANA_VARIANTS=\"${variants}\"")
    list(TRANSFORM NANA_VARIANTS PREPEND "nana_tbp-" OUTPUT_VARIABLE variant_targets)
    add_dependencies(nana_tbp ${variant_targets})
    set_property(TARGET nana_tbp PROPERTY CXX_STANDARD 23)
else()
    nana_add_executable(nana_tbp nana_core "tbp.cpp")
    target_compile_definitions(nana_tbp PRIVATE TBP TBP2)
endif()

if(NANA_GUI AND UNIX AND NOT APPLE)
    target_include_directories(nana_interactive PRIVATE ${X11_INCLUDE_DIR} ${OPENGL_INCLUDE_DIRS} ${PNG_INCLUDE_DIR})
    target_link_libraries(nana_interactive ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${PNG_LIBRARY})
endif()
