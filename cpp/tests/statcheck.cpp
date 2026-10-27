#include "Tetris/game/MarkovOpponent.hpp"
#include "Tetris/game/OpponentFilter.hpp"
#include "Tetris/bot/UCT.hpp"
#include "Tetris/bot/Config.hpp"
#include "Util/json.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(!cc_search, "statcheck tests suffix expectimax");
static_assert(!NANA_FUTURE_ORACLE, "statcheck requires sampled chance");
using Json = nlohmann::json;
constexpr int K = markov::EMISSIONS;
using Prob = std::array<double, K>;
Json results = Json::array();
int failures = 0;

void record(const std::string& name, bool ok, Json detail = Json::object()) {
    detail["name"] = name;
    detail["status"] = ok ? "pass" : "fail";
    failures += !ok;
    results.push_back(detail);
}
void diagnostic(const std::string& name, Json detail) {
    detail["name"] = name;
    detail["status"] = "diagnostic";
    results.push_back(detail);
}
void histogram(const std::string& name, const std::vector<double>& p,
               const std::vector<int>& counts, int n) {
    constexpr double family_alpha = 0.001;
    const double logtail = std::log(2 * 10000 / family_alpha);
    double tv = 0, worst = 0, mass = 0;
    bool ok = true;
    for (size_t i = 0; i < p.size(); ++i) {
        mass += p[i];
        const double error = std::abs(double(counts[i]) / n - p[i]);
        const double bound = std::sqrt(2 * p[i] * (1-p[i]) * logtail / n) + 2*logtail/(3*n);
        worst = std::max(worst, error / bound);
        tv += error / 2;
        ok = ok && error <= bound;
    }
    record(name, ok && std::abs(mass-1) < 1e-5,
           {{"samples",n},{"total_variation",tv},{"max_error_over_bound",worst},{"reference_mass",mass}});
}
int emission(const MarkovOpponent& o) {
    if (o.dead) return markov::DEATH;
    for (int e = 0; e < markov::ATTACKS; ++e)
        if (std::abs(o.last_attack - markov::params().attack_value[e]) < 1e-6) return e;
    throw std::runtime_error("cannot identify emitted attack bin");
}
Prob emitted(const std::array<float, markov::STATES>& row) {
    Prob p{};
    for (int z = 0; z < markov::STATES; ++z)
        for (int e = 0; e < K; ++e)
            p[e] += double(markov::probability(row,z)) * markov::probability(markov::params().emission[z],e);
    return p;
}
std::vector<double> vec(const Prob& p) { return {p.begin(),p.end()}; }
uint32_t trialSeed(uint64_t trial, uint64_t stream) {
    return uint32_t(fast_seed_detail::mix(trial + 0x9e3779b97f4a7c15ULL * (stream+1)) >> 32);
}

void samplerTests(int n, uint64_t seed) {
    const auto& p = markov::params();
    for (int context = 0; context < 13; ++context) {
        const bool fresh = context == 0;
        const int z0 = (context*3) % markov::STATES;
        const int s0 = (context*7) % markov::SINCE;
        const int input1 = context % markov::ACTIONS;
        const int input2 = (context*2+1) % markov::ACTIONS;
        const float meter0 = float((context % 3)*8);
        RootOpponent root;
        root.fresh = fresh;
        root.state[z0] = 1;
        root.since[s0] = 1;
        root.meter = meter0;
        root.last_attack = 1;
        root.publish();
        const int j1 = markov::jeopardy_bucket(meter0+input1);
        const auto& row = fresh ? p.initial : p.transition[input1][j1][s0][z0];
        const Prob first = emitted(row);
        std::vector<double> pair(K*K,0);
        for (int z1=0; z1<markov::STATES; ++z1) {
            for (int e1=0; e1<K; ++e1) {
                const double w = double(markov::probability(row,z1)) * markov::probability(p.emission[z1],e1);
                const bool dead = e1 == markov::DEATH;
                const float a = dead ? 0 : p.attack_value[e1];
                float m = meter0+input1;
                if (!dead) {
                    if (a > 0) m = std::max(0.0f,m-a);
                    else if (!p.stadium_rules) m -= std::min(m,markov::GARBAGE_CAP);
                }
                m += input2;
                if (p.stadium_rules && a <= 0) m = 0;
                const int s1 = e1 == 0 ? std::min((fresh ? 0 : s0)+1,markov::SINCE-1) : 0;
                const auto& next = dead ? p.initial : p.transition[input2][markov::jeopardy_bucket(m)][s1][z1];
                const Prob second = emitted(next);
                for (int e2=0; e2<K; ++e2) pair[e1*K+e2] += w*second[e2];
            }
        }
        std::vector<int> c1(K,0), c2(K*K,0);
        bool cancellation = true;
        for (int i=0; i<n; ++i) {
            MarkovOpponent o;
            o.adopt(root);
            o.seed(trialSeed(i,seed+context));
            o.resample_from_root();
            o.receiveAttack(input1);
            const double m = o.meter;
            const double sent = o.play();
            const int a = emission(o);
            const double expected_sent = a == markov::DEATH ? 0 : std::max(0.0,double(p.attack_value[a])-m);
            cancellation = cancellation && std::abs(sent-expected_sent)<1e-5;
            o.receiveAttack(input2);
            o.play();
            const int b = emission(o);
            ++c1[a]; ++c2[a*K+b];
        }
        histogram("sampler/one-turn/"+std::to_string(context),vec(first),c1,n);
        histogram("sampler/two-turn-joint/"+std::to_string(context),pair,c2,n);
        record("sampler/cancellation/"+std::to_string(context),cancellation);
    }
    markov::root_belief.active = false;
    fast_seed_reset(seed);
    std::vector<int> counts(K,0);
    MarkovOpponent base;
    for (int i=0; i<n; ++i) {
        auto o = base;
        o.reset_rng(); o.play(); ++counts[emission(o)];
    }
    histogram("sampler/reseed-fresh-root",vec(emitted(p.initial)),counts,n);
    const Prob marginal=emitted(p.initial);
    std::vector<double> independent(K*K,0);
    std::vector<int> successive(K*K,0);
    for(int a=0;a<K;++a) for(int b=0;b<K;++b) independent[a*K+b]=marginal[a]*marginal[b];
    for(int i=0;i<n;++i) {
        auto a=base, b=base;
        a.reset_rng(); a.play(); b.reset_rng(); b.play();
        ++successive[emission(a)*K+emission(b)];
    }
    histogram("sampler/successive-reseed-independence",independent,successive,n);
}

Return value(double x) { return Return{0,0,float(x/config().eval_weight),1,0}; }
double mean(const ChildTable& t) {
    float n=0;
    const auto s=t.sum(n);
    return rewardScalar(scaled(s,1/n));
}
UCTNode node(int actions) {
    UCTNode n;
    n.id=1; n.N=0;
    n.edge_stats.resize(actions); n.cold.resize(actions);
    return n;
}
void backupTests(int samples, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (int k : {2,8,17}) {
        std::vector<double> weights(k), values(k);
        double total=0, target=0;
        for (int i=0;i<k;++i) { weights[i]=i+1; values[i]=double(i)/(k-1); total+=weights[i]; }
        for (int i=0;i<k;++i) target+=weights[i]*values[i]/total;
        std::discrete_distribution<int> draw(weights.begin(),weights.end());
        ChildTable t;
        for (int i=0;i<samples;++i) { const int c=draw(rng); t.put(c+1,value(values[c]),1); }
        const double bound=std::sqrt(std::log(2*10000/0.001)/(2*samples));
        record("backup/fixed-values/"+std::to_string(k),std::abs(mean(t)-target)<bound,
               {{"estimate",mean(t)},{"exact",target},{"bound",bound}});
    }
    auto n=node(2);
    n.report(0,11,value(0),3,false); n.report(0,12,value(1),1,false);
    n.report(1,21,value(0.4),4,false);
    record("backup/tiny-tree-best-action",n.best==1 && std::abs(rewardScalar(n.value())-0.4)<1e-6);
    n.report(0,11,value(0.8),1,false);
    record("backup/latest-child-propagates",n.best==0 && std::abs(rewardScalar(n.value())-0.84)<1e-5);
    n.report(0,11,value(0),1,false);
    record("backup/decreasing-best-rescans",n.best==1 && std::abs(rewardScalar(n.value())-0.4)<1e-6);

    ChildTable draws, effort;
    for (int i=0;i<1000;++i) {
        const bool rare=i%10==0;
        draws.put(rare?2:1,value(rare?1:0),1);
        effort.put(rare?2:1,value(rare?1:0),rare?10:1);
    }
    diagnostic("backup/outcome-dependent-batch-weights",
        {{"true_chance_mean",0.1},{"one_count_per_draw",mean(draws)},
         {"ten_counts_for_rare_child",mean(effort)},
         {"interpretation","If batch k counts descendant work rather than fresh chance draws, probabilities become effort weighted. Full scheduler needs a separate audit."}});
    ChildTable cached;
    for (int c=1;c<=9;++c) cached.put(c,value(0.1),1);
    for (int c=1;c<=9;++c) cached.put(c,value(0.9),1);
    diagnostic("backup/overflow-stale-values",{{"all_latest_values",0.9},{"actual",mean(cached)},
        {"interpretation","First eight children replace prior values; overflow children retain historical returns."}});
    ChildTable merged;
    for (int i=0;i<1000;++i) merged.put(7,value(i%2),1);
    diagnostic("backup/merged-child-last-sample",{{"mean_of_samples",0.5},{"actual",mean(merged)},
        {"interpretation","A child hash must identify a stable estimated value. State-specific samples under one hash are replaced, not averaged."}});
    const Return short_path{1,1,0,1,0}, long_path{0,9,0,1,0};
    const double pooled=rewardScalar(scaled(short_path+long_path,0.5))/config().app_weight;
    diagnostic("reward/ratio-of-means",{{"mean_of_path_rates",0.5},{"pooled_rate",pooled},
        {"interpretation","Suffix reward uses pooled attack/length, not the expected per-path attack rate. This is an objective choice requiring explicit interpretation."}});
}

void rootDiagnostic(int samples, uint64_t seed, int hidden_turns) {
    OpponentFilter f;
    for(int t=0;t<hidden_turns;++t) f.turn(0,true,0);
    const auto root=f.root();
    const auto exact=f.predicted();
    std::array<double,markov::STATES> product{};
    const auto& p=markov::params();
    for (int z=0;z<markov::STATES;++z)
        for (int s=0;s<markov::SINCE;++s)
            for (int to=0;to<markov::STATES;++to)
                product[to]+=double(root.state[z])*root.since[s]*markov::probability(p.transition[0][0][s][z],to);
    Prob target{}, joint_target{};
    double state_tv=0;
    for (int z=0;z<markov::STATES;++z) {
        state_tv+=std::abs(product[z]-exact[z])/2;
        for (int e=0;e<K;++e) {
            target[e]+=product[z]*markov::probability(p.emission[z],e);
            joint_target[e]+=exact[z]*markov::probability(p.emission[z],e);
        }
    }
    root.publish();
    fast_seed_reset(seed+hidden_turns);
    EmulationGame base(false);
    base.set_tetrio_rules();
    base.reset_opponent();
    base.markov.adopt(root);
    std::vector<int> counts(K,0);
    for (int i=0;i<samples;++i) {
        auto o=base; o.start_from_root(); o.chance_move(); ++counts[emission(o.markov)];
    }
    histogram("root/marginal-sampler-contract/"+std::to_string(hidden_turns),vec(target),counts,samples);
    double attack_tv=0;
    for (int e=0;e<K;++e) attack_tv+=std::abs(target[e]-joint_target[e])/2;
    diagnostic("root/joint-to-product-approximation/"+std::to_string(hidden_turns),{{"hidden_turns",hidden_turns},{"next_state_total_variation",state_tv},
        {"next_emission_total_variation",attack_tv},{"joint_emission",vec(joint_target)},{"product_emission",vec(target)},
        {"interpretation","The filter tracks joint (state,since), but the search root samples their marginals independently."}});
    markov::root_belief.active=false;
}
int main(int argc, char** argv) {
    try {
        const int n=argc>1 ? std::stoi(argv[1]) : 100000;
        const uint64_t seed=argc>2 ? std::stoull(argv[2]) : 20261024;
        if(n<10000) throw std::runtime_error("use at least 10000 trials");
        fast_seed_reset(seed);
        samplerTests(n,seed);
        backupTests(n,seed);
        for(int turns : {1,8,24,64}) rootDiagnostic(n,seed,turns);
        std::cout << Json{{"trials_per_histogram",n},{"seed",seed},{"family_alpha",0.001},
                          {"comparison_budget",10000},{"failures",failures},{"results",results}}.dump(2) << '\n';
        return failures ? 1 : 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
