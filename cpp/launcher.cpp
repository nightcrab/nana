// launcher.cpp : tbp_tetris when Nana is built with NANA_VARIANTS. Nana is built once per ISA level
// (tbp_tetris-znver5, tbp_tetris-x86-64-v3, ...) and this runs the best one the CPU supports, handing
// it our stdin / stdout and arguments, so TBP clients see one bot. It is built for the compiler's
// default level so it runs anywhere.
//
//   NANA_VARIANT=<level>          run that level whatever the CPU says
//   tbp_tetris --list-variants    show what each built level needs that the CPU lacks, and the pick

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <climits>
#include <unistd.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#ifndef NANA_VARIANTS
#error "NANA_VARIANTS lists the built levels, set by CMake"
#endif

namespace {

enum Reg { EAX, EBX, ECX, EDX };

struct CpuBit {
    const char* name;
    unsigned leaf, subleaf;
    Reg reg;
    int bit;
};

// the CPUID bits behind the features the levels below need
constexpr CpuBit cpu_bits[] = {
    {"sse2", 1, 0, EDX, 26},
    {"sse3", 1, 0, ECX, 0},
    {"pclmul", 1, 0, ECX, 1},
    {"ssse3", 1, 0, ECX, 9},
    {"fma", 1, 0, ECX, 12},
    {"cx16", 1, 0, ECX, 13},
    {"sse4.1", 1, 0, ECX, 19},
    {"sse4.2", 1, 0, ECX, 20},
    {"movbe", 1, 0, ECX, 22},
    {"popcnt", 1, 0, ECX, 23},
    {"aes", 1, 0, ECX, 25},
    {"xsave", 1, 0, ECX, 26},
    {"avx", 1, 0, ECX, 28},
    {"f16c", 1, 0, ECX, 29},
    {"bmi", 7, 0, EBX, 3},
    {"avx2", 7, 0, EBX, 5},
    {"bmi2", 7, 0, EBX, 8},
    {"avx512f", 7, 0, EBX, 16},
    {"avx512dq", 7, 0, EBX, 17},
    {"adx", 7, 0, EBX, 19},
    {"avx512ifma", 7, 0, EBX, 21},
    {"avx512cd", 7, 0, EBX, 28},
    {"avx512bw", 7, 0, EBX, 30},
    {"avx512vl", 7, 0, EBX, 31},
    {"avx512vbmi", 7, 0, ECX, 1},
    {"avx512vbmi2", 7, 0, ECX, 6},
    {"gfni", 7, 0, ECX, 8},
    {"vaes", 7, 0, ECX, 9},
    {"vpclmulqdq", 7, 0, ECX, 10},
    {"avx512vnni", 7, 0, ECX, 11},
    {"avx512bitalg", 7, 0, ECX, 12},
    {"avx512vpopcntdq", 7, 0, ECX, 14},
    {"avx512vp2intersect", 7, 0, EDX, 8},
    {"avxvnni", 7, 1, EAX, 4},
    {"avx512bf16", 7, 1, EAX, 5},
    {"lahf_lm", 0x80000001, 0, ECX, 0},
    {"lzcnt", 0x80000001, 0, ECX, 5},
    {"sse4a", 0x80000001, 0, ECX, 6},
    {"prfchw", 0x80000001, 0, ECX, 8},
};

// what each level needs, best first. the x86-64 levels are the psABI's; for znver4 / znver5 it is
// what the compiler may emit on its own (the AVX-512 extensions, GFNI, VNNI, ...). the rest they
// enable (rdrnd, movdiri, clwb, ...) only comes from intrinsics, which Nana doesn't use.
// os-avx / os-avx512: the OS saves the AVX / AVX-512 registers (XCR0)
#define NEEDS_V1 "sse2"
#define NEEDS_V2 NEEDS_V1 " cx16 lahf_lm popcnt sse3 ssse3 sse4.1 sse4.2"
#define NEEDS_V3 NEEDS_V2 " os-avx avx avx2 bmi bmi2 f16c fma lzcnt movbe xsave"
#define NEEDS_V4 NEEDS_V3 " os-avx512 avx512f avx512bw avx512cd avx512dq avx512vl"
#define NEEDS_ZEN4 NEEDS_V4 " adx aes avx512bf16 avx512bitalg avx512ifma avx512vbmi avx512vbmi2" \
    " avx512vnni avx512vpopcntdq gfni pclmul prfchw sse4a vaes vpclmulqdq"
#define NEEDS_ZEN5 NEEDS_ZEN4 " avx512vp2intersect avxvnni"

struct Level {
    const char* name;
    const char* needs;
};

constexpr Level levels[] = {
    {"znver5", NEEDS_ZEN5},
    {"znver4", NEEDS_ZEN4},
    {"x86-64-v4", NEEDS_V4},
    {"x86-64-v3", NEEDS_V3},
    {"x86-64-v2", NEEDS_V2},
    {"x86-64", NEEDS_V1},
};

void cpuid(unsigned leaf, unsigned subleaf, unsigned out[4]) {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, int(leaf), int(subleaf));
    for (int i = 0; i < 4; ++i) {
        out[i] = unsigned(regs[i]);
    }
#else
    __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
#endif
}

unsigned long long xgetbv0() {
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#else
    // clang's _xgetbv wants the whole file built with xsave; the launcher is built without it
    unsigned lo, hi;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(0));
    return (unsigned long long)hi << 32 | lo;
#endif
}

std::vector<std::string> split(const std::string& text, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= text.size()) {
        size_t end = text.find(sep, start);
        if (end == std::string::npos) {
            end = text.size();
        }
        if (end > start) {
            out.push_back(text.substr(start, end - start));
        }
        start = end + 1;
    }
    return out;
}

bool contains(const std::vector<std::string>& list, const std::string& item) {
    for (const auto& x : list) {
        if (x == item) {
            return true;
        }
    }
    return false;
}

// the features this CPU and OS have, by the names above
std::vector<std::string> detect() {
    std::vector<std::string> have;
    unsigned regs[4];
    cpuid(0, 0, regs);
    const unsigned max_leaf = regs[EAX];
    cpuid(0x80000000, 0, regs);
    const unsigned max_ext_leaf = regs[EAX];
    unsigned max_subleaf7 = 0;
    if (max_leaf >= 7) {
        cpuid(7, 0, regs);
        max_subleaf7 = regs[EAX];
    }

    for (const CpuBit& b : cpu_bits) {
        const bool valid = b.leaf >= 0x80000000 ? b.leaf <= max_ext_leaf
                                                : b.leaf <= max_leaf && (b.leaf != 7 || b.subleaf <= max_subleaf7);
        if (!valid) {
            continue;
        }
        cpuid(b.leaf, b.subleaf, regs);
        if (regs[b.reg] >> b.bit & 1) {
            have.push_back(b.name);
        }
    }

    cpuid(1, 0, regs);
    const bool osxsave = regs[ECX] >> 27 & 1;
    if (osxsave) {
        const unsigned long long xcr0 = xgetbv0();
        // SSE and AVX state; then opmask and both halves of zmm
        if ((xcr0 & 0x6) == 0x6) {
            have.push_back("os-avx");
        }
        if ((xcr0 & 0xe6) == 0xe6) {
            have.push_back("os-avx512");
        }
    }
    return have;
}

std::vector<std::string> missing(const Level& level, const std::vector<std::string>& have) {
    std::vector<std::string> out;
    for (const auto& need : split(level.needs, ' ')) {
        if (!contains(have, need)) {
            out.push_back(need);
        }
    }
    return out;
}

const Level* findLevel(const std::string& name) {
    for (const Level& level : levels) {
        if (name == level.name) {
            return &level;
        }
    }
    return nullptr;
}

#ifdef _WIN32

std::wstring widen(const std::string& ascii) {
    return std::wstring(ascii.begin(), ascii.end());
}

std::wstring selfPath() {
    std::wstring path(MAX_PATH, L'\0');
    while (true) {
        const DWORD n = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
        if (n < path.size()) {
            path.resize(n);
            return path;
        }
        path.resize(path.size() * 2);
    }
}

// our command line after the program name, verbatim, as CommandLineToArgvW splits off argv[0]
const wchar_t* afterProgramName(const wchar_t* cmd) {
    if (*cmd == L'"') {
        ++cmd;
        while (*cmd && *cmd != L'"') {
            ++cmd;
        }
        if (*cmd) {
            ++cmd;
        }
    }
    else {
        while (*cmd && *cmd != L' ' && *cmd != L'\t') {
            ++cmd;
        }
    }
    return cmd;
}

int run(const std::string& level) {
    std::wstring self = selfPath();
    const size_t slash = self.find_last_of(L"\\/");
    const std::wstring dir = self.substr(0, slash + 1);
    std::wstring stem = self.substr(slash + 1);
    if (stem.size() > 4 && _wcsicmp(stem.c_str() + stem.size() - 4, L".exe") == 0) {
        stem.resize(stem.size() - 4);
    }
    const std::wstring path = dir + stem + L"-" + widen(level) + L".exe";
    std::wstring cmd = L"\"" + path + L"\"" + afterProgramName(GetCommandLineW());

    // the child dies with us: TBP clients stop the bot by killing the process they started
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
    }

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    for (HANDLE h : {startup.hStdInput, startup.hStdOutput, startup.hStdError}) {
        if (h && h != INVALID_HANDLE_VALUE) {
            SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
        }
    }

    PROCESS_INFORMATION child{};
    if (!CreateProcessW(path.c_str(), cmd.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr,
                        &startup, &child)) {
        std::fprintf(stderr, "nana launcher: can't start %ls (error %lu)\n", path.c_str(), GetLastError());
        return 127;
    }
    if (job) {
        AssignProcessToJobObject(job, child.hProcess);
    }
    ResumeThread(child.hThread);
    CloseHandle(child.hThread);

    // Ctrl+C reaches the child too, which decides what to do with it
    SetConsoleCtrlHandler(nullptr, TRUE);
    WaitForSingleObject(child.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(child.hProcess, &code);
    return int(code);
}

#else

int run(const std::string& level, char** argv) {
    std::string self;
    char buf[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        self.assign(buf, size_t(n));
    }
    else {
        self = argv[0];
    }
    const size_t slash = self.find_last_of('/');
    const std::string path = self.substr(0, slash + 1) + self.substr(slash + 1) + "-" + level;
    execv(path.c_str(), argv);
    std::fprintf(stderr, "nana launcher: can't start %s: %s\n", path.c_str(), std::strerror(errno));
    return 127;
}

#endif

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> built = split(NANA_VARIANTS, ',');
    const std::vector<std::string> have = detect();

    const Level* pick = nullptr;
    for (const Level& level : levels) {
        if (contains(built, level.name) && missing(level, have).empty()) {
            pick = &level;
            break;
        }
    }

    if (argc > 1 && std::strcmp(argv[1], "--list-variants") == 0) {
        for (const Level& level : levels) {
            if (!contains(built, level.name)) {
                continue;
            }
            const auto lacks = missing(level, have);
            std::printf("%-10s %s", level.name, lacks.empty() ? "supported" : "lacks");
            for (const auto& f : lacks) {
                std::printf(" %s", f.c_str());
            }
            std::printf("\n");
        }
        std::printf("runs: %s\n", pick ? pick->name : "none");
        return 0;
    }

    if (const char* forced = std::getenv("NANA_VARIANT"); forced && *forced) {
        if (!contains(built, forced)) {
            std::fprintf(stderr, "nana launcher: NANA_VARIANT=%s isn't built; built: %s\n", forced, NANA_VARIANTS);
            return 1;
        }
        pick = findLevel(forced);
        if (!missing(*pick, have).empty()) {
            std::fprintf(stderr, "nana launcher: warning: this CPU doesn't support %s\n", forced);
        }
    }

    if (!pick) {
        std::fprintf(stderr, "nana launcher: this CPU supports none of the built levels (%s)\n", NANA_VARIANTS);
        return 1;
    }

#ifdef _WIN32
    return run(pick->name);
#else
    return run(pick->name, argv);
#endif
}
