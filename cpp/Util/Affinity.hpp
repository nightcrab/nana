#pragma once

void pinThreadToCore(int index);

int physicalCoreCount();

int cacheGroupOfCore(int index);

bool restrictProcessToCpu(int cpu);
