#include "Affinity.hpp"

#include <algorithm>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#endif

namespace {

struct LogicalCpu {
    int group = 0;  // processor group (Windows only)
    int number = 0; // cpu number within the group
};

struct Topology {
    // logical cpus in physical-core-first order
    std::vector<LogicalCpu> order;
    int cores = 0;
};

// cores[i] holds core i's logical cpus; interleave them into physical-core-first order
Topology interleave(const std::vector<std::vector<LogicalCpu>>& cores) {
    Topology topology;
    topology.cores = (int)cores.size();
    size_t widest = 0;
    for (const auto& core : cores) {
        widest = std::max(widest, core.size());
    }
    for (size_t sibling = 0; sibling < widest; sibling++) {
        for (const auto& core : cores) {
            if (sibling < core.size()) {
                topology.order.push_back(core[sibling]);
            }
        }
    }
    return topology;
}

#ifdef _WIN32

Topology readTopology() {
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    std::vector<char> buffer(length);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
    if (length == 0 || !GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
        return {};
    }

    // only use cpus the process is allowed on (group 0 masks; fine below 64 cpus)
    DWORD_PTR process_mask = 0, system_mask = 0;
    GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask);

    std::vector<std::vector<LogicalCpu>> cores;
    for (DWORD offset = 0; offset < length;) {
        auto* entry = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        std::vector<LogicalCpu> core;
        for (WORD g = 0; g < entry->Processor.GroupCount; g++) {
            const GROUP_AFFINITY& affinity = entry->Processor.GroupMask[g];
            for (int bit = 0; bit < (int)sizeof(KAFFINITY) * 8; bit++) {
                KAFFINITY cpu = KAFFINITY(1) << bit;
                if (!(affinity.Mask & cpu)) continue;
                if (affinity.Group == 0 && process_mask && !(process_mask & cpu)) continue;
                core.push_back({ affinity.Group, bit });
            }
        }
        if (!core.empty()) {
            cores.push_back(core);
        }
        offset += entry->Size;
    }
    return interleave(cores);
}

void pin(const LogicalCpu& cpu) {
    GROUP_AFFINITY affinity = {};
    affinity.Group = (WORD)cpu.group;
    affinity.Mask = KAFFINITY(1) << cpu.number;
    SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr);
}

#else

// "0,16" or "0-1" style cpu lists
std::vector<int> parseCpuList(const std::string& list) {
    std::vector<int> cpus;
    std::stringstream ss(list);
    std::string part;
    while (std::getline(ss, part, ',')) {
        size_t dash = part.find('-');
        int first = std::stoi(part.substr(0, dash));
        int last = dash == std::string::npos ? first : std::stoi(part.substr(dash + 1));
        for (int c = first; c <= last; c++) cpus.push_back(c);
    }
    return cpus;
}

Topology readTopology() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return {};
    }

    // group allowed cpus by their sibling list, which identifies the physical core
    std::map<std::string, std::vector<LogicalCpu>> by_core;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        std::string siblings;
        if (!(file >> siblings)) {
            siblings = std::to_string(cpu);
        }
        by_core[siblings].push_back({ 0, cpu });
    }

    std::vector<std::vector<LogicalCpu>> cores;
    for (auto& [siblings, core] : by_core) {
        cores.push_back(core);
    }
    // keep cores in cpu order, so worker 0 is on the lowest core
    std::sort(cores.begin(), cores.end(), [](const auto& a, const auto& b) { return a[0].number < b[0].number; });
    return interleave(cores);
}

void pin(const LogicalCpu& cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu.number, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

#endif

const Topology& topology() {
    static const Topology topology = readTopology();
    return topology;
}

} // namespace

void pinThreadToCore(int index) {
    const Topology& t = topology();
    if (t.order.empty()) {
        return;
    }
    pin(t.order[index % t.order.size()]);
}

int physicalCoreCount() {
    return topology().cores;
}
