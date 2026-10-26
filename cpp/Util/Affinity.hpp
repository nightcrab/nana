#pragma once

#include <vector>

void pinThreadToCore(int index);

int physicalCoreCount();

int cacheGroupOfCore(int index);

bool restrictProcessToCpu(int cpu);
bool restrictProcessToCpus(const std::vector<int>& cpus);
