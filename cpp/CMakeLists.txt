# CMakeList.txt : CMake project for nana, include source and define
# project specific logic here.
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Tetris)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Tetris/game)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Tetris/bot)

# only for its source list: the engine cores below compile ShakTris themselves, once per ISA level
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
	"Tetris/game/Opponent.cpp"
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

# the graphical nana_interactive needs X11, OpenGL and libpng on Linux;
# turn it off to build only the bots and benches on a headless machine
option(NANA_GUI "Build the graphical nana_interactive" ON)

if(NANA_GUI AND UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(PNG REQUIRED)
endif()


# Nana is built for an ISA level picked by -march (native, x86-64, x86-64-v3, znver5, ...).
# NANA_MARCH is the level of the benches, the emulator and the interactive build.
# NANA_VARIANTS builds tbp_tetris once per listed level, as tbp_tetris-<level>, and makes tbp_tetris
# a launcher that runs the best one the CPU supports (launcher.cpp). For releases, e.g.
# -DNANA_VARIANTS="znver5;x86-64-v3;x86-64"
set(NANA_MARCH "native" CACHE STRING "-march for the benches, emulator and interactive build")
set(NANA_VARIANTS "" CACHE STRING "ISA levels to build tbp_tetris for, behind a launcher")
# the levels launcher.cpp can detect
set(NANA_KNOWN_VARIANTS znver5 znver4 x86-64-v4 x86-64-v3 x86-64-v2 x86-64)

# the compile options for an ISA level
function(nana_arch_flags out march)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # cl.exe has no -march, only the nearest /arch
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

# Nana's and ShakTris' sources for one ISA level, linked into each executable that uses them
function(nana_add_core name march)
    add_library(${name} OBJECT ${TETRIS_SOURCES} ${SHAKTRIS_SOURCES})
    nana_arch_flags(flags ${march})
    # public: the executable's own source is built for the same level
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
# the same bench without the NANA_BENCH profiling; prints only simulations / second
nana_add_executable(nana_speed nana_core "bench.cpp")
nana_add_executable(nana_msgbench nana_core "msgbench.cpp")
nana_add_executable(nana_emulate nana_core "emulate.cpp")

if(NANA_VARIANTS)
    foreach(variant IN LISTS NANA_VARIANTS)
        if(NOT variant IN_LIST NANA_KNOWN_VARIANTS)
            message(FATAL_ERROR "NANA_VARIANTS: the launcher can't detect ${variant}; use some of ${NANA_KNOWN_VARIANTS}")
        endif()
        nana_add_core(nana_core_${variant} ${variant})
        nana_add_executable(tbp_tetris-${variant} nana_core_${variant} "tbp.cpp")
        target_compile_definitions(tbp_tetris-${variant} PRIVATE TBP TBP2)
    endforeach()

    # built for the compiler's default level, so it runs anywhere
    add_executable(tbp_tetris "launcher.cpp" "Tetris/bot/Config.cpp" ${NANA_ICON})
    list(JOIN NANA_VARIANTS "," variants)
    target_compile_definitions(tbp_tetris PRIVATE "NANA_VARIANTS=\"${variants}\"")
    list(TRANSFORM NANA_VARIANTS PREPEND "tbp_tetris-" OUTPUT_VARIABLE variant_targets)
    add_dependencies(tbp_tetris ${variant_targets})
    set_property(TARGET tbp_tetris PROPERTY CXX_STANDARD 23)
else()
    nana_add_executable(tbp_tetris nana_core "tbp.cpp")
    target_compile_definitions(tbp_tetris PRIVATE TBP TBP2)
endif()

if(NANA_GUI AND UNIX AND NOT APPLE)
    target_include_directories(nana_interactive PRIVATE ${X11_INCLUDE_DIR} ${OPENGL_INCLUDE_DIRS} ${PNG_INCLUDE_DIR})
    target_link_libraries(nana_interactive ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${PNG_LIBRARY})
endif()
