#pragma once
#include <filesystem>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

inline std::filesystem::path executableDirectory() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(256);
    for (;;) {
        const auto size = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (!size) throw std::runtime_error("cannot locate executable");
        if (size < buffer.size()) return std::filesystem::path(std::wstring(buffer.data(), size)).parent_path();
        buffer.resize(buffer.size() * 2);
    }
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size)) throw std::runtime_error("cannot locate executable");
    return std::filesystem::canonical(buffer.data()).parent_path();
#else
    std::vector<char> buffer(256);
    for (;;) {
        const auto size = readlink("/proc/self/exe", buffer.data(), buffer.size());
        if (size < 0) throw std::runtime_error("cannot locate executable");
        if (static_cast<size_t>(size) < buffer.size())
            return std::filesystem::path(std::string(buffer.data(), size)).parent_path();
        buffer.resize(buffer.size() * 2);
    }
#endif
}
