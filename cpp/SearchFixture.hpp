#pragma once
// Only the optional searchcheck core includes this adapter. Scheduling, selection,
// restart replay, virtual loss, message handling, and backups remain production code.
#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
class EmulationGame;
class UCTNode;
namespace search_fixture {
struct Outcome { int node; double probability; };
using Action = std::vector<Outcome>;
struct Node { std::vector<Action> actions; double payoff = 0; double heuristic = 0; };
struct Model { std::string name; std::vector<Node> nodes; };
inline const Model* model = nullptr;
inline bool inject_weights = false;
inline std::array<std::array<std::atomic<uint64_t>,32>,2> proposals{};
constexpr uint32_t hash_base = 0xe0000000u;
void initialize(UCTNode& node, const EmulationGame& state);
void advance(EmulationGame& state);
double exact(int node);
}
