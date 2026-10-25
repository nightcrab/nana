#include "SearchFixture.hpp"
#include "Tetris/bot/UCT.hpp"
#include "Tetris/bot/Config.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace search_fixture {
void initialize(UCTNode& n, const EmulationGame& state) {
    const auto& fixture=model->nodes.at(state.fixture_node);
    n.id=state.hash(); n.N=1;
    n.edge_stats.resize(fixture.actions.size());
    n.cold.resize(fixture.actions.size());
    n.leaf_V=Return{0,0,float(fixture.heuristic/config().eval_weight),1,0};
    for(size_t i=0;i<fixture.actions.size();++i) {
        Piece p(PieceType::T);
        p.position.x=int(i);
        n.cold[i].move=PackedMove::pack(Move(p,false));
        n.edge_stats[i].prior=1.0f/fixture.actions.size();
    }
}
void advance(EmulationGame& state) {
    const auto& choices=model->nodes.at(state.fixture_node).actions.at(state.fixture_action);
    const double u=state.rng.getRand(65536)/65536.0;
    double cdf=0;
    size_t selected=choices.size()-1;
    for(size_t i=0;i<choices.size();++i) {
        cdf+=choices[i].probability;
        if(u<cdf) { selected=i; break; }
    }
    if(state.fixture_node==0)
        proposals.at(state.fixture_action).at(selected).fetch_add(1,std::memory_order_relaxed);
    state.fixture_node=choices[selected].node;
    state.game_over=model->nodes.at(state.fixture_node).actions.empty();
}
double exact(int node) {
    const auto& n=model->nodes.at(node);
    if(n.actions.empty()) return n.payoff;
    double best=-1;
    for(const auto& a:n.actions) {
        double v=0, mass=0;
        for(const auto& o:a) { v+=o.probability*exact(o.node); mass+=o.probability; }
        if(std::abs(mass-1)>1e-9) throw std::runtime_error("invalid fixture probabilities");
        best=std::max(best,v);
    }
    return best;
}
}
