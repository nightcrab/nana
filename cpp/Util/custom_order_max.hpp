#pragma once

#include <ranges>
#include <vector>
#include <tuple>
#include <algorithm>
#include <optional>

template<std::ranges::range R, typename... Fs>
auto custom_order_max(const R& r, Fs... fs) noexcept {
    auto compare_members = [&]<typename... Rest>(auto &self, const auto& a, const auto& b, Rest... rest) {
        if constexpr (sizeof...(Rest) == 0) {
            return false;
        } else {
            auto first_member = std::get<0>(std::forward_as_tuple(rest...));;
             
            if ((a.*first_member) != (b.*first_member)) {
                return a.*first_member < b.*first_member;
            }
            
            return self(self, a, b, rest...);
        }
    };
    
    return std::ranges::empty(r) ? std::nullopt : std::optional{*std::ranges::max_element(r, [&](const auto& a, const auto& b) {
        return compare_members(compare_members, a, b, fs...);
    })};
}