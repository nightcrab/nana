#pragma once
#include "engine/Board.hpp"
#include "engine/Game.hpp"
#include "util/rng.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

enum LayerType :uint8_t {
    SPIN,
    COMBO,
    CLEAN,
    MESSY,
};

inline std::ostream& operator<<(std::ostream& os, const LayerType& _Val) {
    std::string out;
    switch (_Val) {
    case SPIN:
        out = "SPIN";
        break;
    case COMBO:
        out = "COMBO";
        break;
    case CLEAN:
        out = "CLEAN";
        break;
    case MESSY:
        out = "MESSY";
        break;
    default:
        out = "UNKNOWN";
        break;
    }

    os << out;
    return os;
}


enum TacticState :uint8_t {
    BUILD_SPIN,
    BUILD_COMBO,
    BUILD_CLEAN,
    ATTACK,
};

inline std::ostream& operator<<(std::ostream& os, const TacticState& _Val) {
    std::string out;
    switch (_Val) {
    case BUILD_SPIN:
        out = "BUILD_SPIN";
        break;
    case BUILD_COMBO:
        out = "BUILD_COMBO";
        break;
    case BUILD_CLEAN:
        out = "BUILD_CLEAN";
        break;
    case ATTACK:
        out = "ATTACK";
        break;
    default:
        out = "UNKNOWN";
        break;
    }

    os << out;
    return os;
}


class StackLayer {
public:
    double height;
    LayerType type;
    // whether clean below can be accessed
    bool open = false;

    StackLayer() {
        height = 0;
        type = COMBO;
    }
    StackLayer(LayerType type, double height) : height(height), type(type) {}
    StackLayer(LayerType type, double height, bool open) : height(height), type(type), open(open) {}
};

constexpr std::array<double, 20> comboTable = { 0, 1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };

// Probability of layers behaving as MESSY during attack, expressed as percentages.
constexpr double MISTAKE_PROB_CLEAN = 5;
constexpr double MISTAKE_PROB_COMBO = 5;
constexpr double MISTAKE_PROB_SPIN = 5;

constexpr double MISTAKE_DOWNSTACK_PROB = 5;

constexpr double WASTE_I_PROB = 10;

constexpr double COMBO_GARBAGE_PROB = 40;

// Probability the opponent will attack, if it can.
constexpr double ATTACK_PROB = 80;

// Probability of a spin exposing the lower garbage well
constexpr double OPEN_SPIN_PROB = 40;

class Opponent {
public:
    Opponent (const Game& game);

    Opponent() = default;
    // Copy constructor
    Opponent(const Opponent& other) noexcept = default;
    // Move constructor
    Opponent(Opponent&& other) noexcept = default;
    ~Opponent() noexcept = default;
    Opponent& operator=(const Opponent& other) noexcept = default;

    RNG rng;
    bool hasI = true;
    bool hasSpinPiece = true;
    bool dead = false;
    TacticState state;


    // Abstraction of the board.
    std::vector<StackLayer> stack = { { CLEAN, 4 } };

    // Represents the incoming garbage meter.
    std::vector<StackLayer> garbage;

    double deaths = 0;

    float pieces = 0;
    float total_damage = 0;

    double nextI = rng.getRand(7);;
    double nextSpinPiece = rng.getRand(7);

    double combo = 0;

    // these are in here because otherwise the linker has a fit
    std::string toString(LayerType layer) {
        switch (layer) {
        case SPIN: return "SPIN";
        case COMBO: return "COMBO";
        case CLEAN: return "CLEAN";
        case MESSY: return "MESSY";
        default: return "UNKNOWN";
        }
    }

    std::string toString(TacticState state) {
        switch (state) {
        case BUILD_SPIN: return "BUILD_SPIN";
        case BUILD_COMBO: return "BUILD_COMBO";
        case BUILD_CLEAN: return "BUILD_CLEAN";
        case ATTACK: return "ATTACK";
        default: return "UNKNOWN";
        }
    }

    std::stringstream stateString() {
        std::stringstream out;
        out << "Stack Height: ";
        out << stack_height();
        out << "\n";
        out << "hasI: ";
        out << hasI;
        out << "\n";
        out << "hasSpinPiece: ";
        out << hasSpinPiece;
        out << "\n";
        out << "nextI: ";
        out << nextI;
        out << "\n";
        out << "nextSpinPiece: ";
        out << nextSpinPiece;
        out << "\n";
        out << "State: ";
        out << toString(state);
        out << "\n";
        out << "Deaths: ";
        out << deaths;
        out << "\n";
        out << "Opponent APP: ";
        out << total_damage / pieces;
        out << "\n";
        for (StackLayer layer : stack) {
            out << toString(layer.type) << ", " << layer.height << "\n";
        }
        return out;
    }

    void reset_rng() {
        rng.new_seed();
    }

    void nextBuild() {
        int rand_int = rng.getRand(100);
        if (rand_int < 50) {
            state = BUILD_SPIN;
            return;
        }
        if (rand_int < 90) {
            state = BUILD_CLEAN;
            return;
        }
        state = BUILD_COMBO;
    }


    void continueBuild() {
        switch (stack.back().type) {
        case SPIN:
            state = BUILD_SPIN;
            break;
        case CLEAN:
            state = BUILD_CLEAN;
            break;
        case COMBO:
            state = BUILD_COMBO;
            break;
        case MESSY:
            state = BUILD_SPIN;
            break;
        default:
            break;
        }
    }

    bool can_attack() {
        if (stack.empty()) {
            return false;
        }
        double height = stack.back().height;
        switch (stack.back().type) {
        case SPIN:
            if (height >= 2) {
                return true;
            }
            break;
        case CLEAN:
            if (height >= 4) {
                return true;
            }
            break;
        case COMBO:
            if (height + combo >= 5) {
                return true;
            }
            break;
        case MESSY:
            if (height >= 1) {
                return true;
            }
            break;
        default:
            break;
        }
        return false;
    }

    bool should_cancel() {
        if (can_attack()) {
            return true;
        }
        if (garbage_height() + stack_height() > 15) {
            return true;
        }
        return false;
    }

    void nextState() {
        // Markovian state transitions
        if (stack.empty()) {
            nextBuild();
            return;
        }
        if (!can_attack()) {
            continueBuild();
            return;
        }
        else if (should_cancel()) {
            // cancel
            state = ATTACK;
            return;
        }
        else if (stack_height() >= 10) {
            // downstack
            state = ATTACK;
            return;
        }
        else if (stack.back().type == MESSY) {
            // downstack
            state = ATTACK;
            return;
        }
        else if (state == ATTACK && can_attack()) {
            // keep attacking
            return;
        }
        else if (rng.getRand(100) < ATTACK_PROB && can_attack()) {
            if (state != ATTACK) {
                state = ATTACK;
            }
        }
        else {
            double height = stack.back().height;

            // change build only if the current one is done
            switch (stack.back().type) {
            case SPIN:
                if (height >= 2) {
                    nextBuild();
                }
                break;
            case CLEAN:
                if (height >= 4) {
                    nextBuild();
                }
                break;
            case COMBO:
                if (rng.getRand(5) == 0) {
                    nextBuild();
                }
                break;
            default:
                nextBuild();
            }
        }
    }

    void build() {
        switch (state) {
        case BUILD_COMBO:
            buildCombo(); break;
        case BUILD_SPIN:
            buildSpin(); break;
        case BUILD_CLEAN:
            buildClean(); break;
        default:
            buildClean();
        }
    }

    void buildSpin() {
        if (stack.empty()) {

            // Start building spin
            bool open = rng.getRand(100) < OPEN_SPIN_PROB;
            stack.push_back({ SPIN, 0.45, open });
            return;

        }

        if (stack.back().type == SPIN) {

            // Continue current build
            stack.back().height += 0.45;

        } else {

            // Round off the current layer
            stack.back().height = std::floor(stack.back().height);

            // Start building spin
            bool open = rng.getRand(100) < OPEN_SPIN_PROB;
            stack.push_back({ SPIN, 0.45, open });
        }

    }

    void buildCombo() {
        if (stack.empty()) {

            stack.push_back({ COMBO, 0.5 });
            return;

        }

        if (stack.back().type == COMBO) {

            // Continue current build
            stack.back().height += 0.5;

        } else {

            // Round off the current layer
            stack.back().height = std::floor(stack.back().height);

            double leftover = 0;

            double height = stack.back().height;

            // Combo stack can borrow excess minos from the layer underneath
            leftover = height - (double)height;

            // Start building combo
            stack.push_back({ COMBO, 0.5 + leftover });

        }
    }

    void buildClean() {
        if (stack.empty()) {
            stack.push_back({ CLEAN, 0.4 });
            return;

        }

        if (stack.back().type == CLEAN) {

            // Continue current build
            stack.back().height += 0.6;

        }
        else {

            // Round off the current layer
            stack.back().height = std::floor(stack.back().height);

            double leftover = 0;

            double height = stack.back().height;

            // Combo stack can borrow excess minos from the layer underneath
            leftover = height - (double)height;

            // Start building clean
            stack.push_back({ CLEAN, 0.6 + leftover, true});

        }
    }

    void attackSpin(double& damage, StackLayer& topLayer) {
        if (topLayer.height < 2) {
            //topLayer.type = COMBO;
            //attackCombo(damage, topLayer);
            return;
        }
        if (!hasSpinPiece) {
            // required piece not available
            damage = 0;

            // we may be able to attack using the clean layer underneath
            if (topLayer.open && hasI) {
                if (stack.size() > 1) {
                    StackLayer secondTopLayer = stack[stack.size() - 2];
                    attackClean(damage, secondTopLayer);
                }
            }

            return;
        }
        damage = 4;
        stack.pop_back();

        hasSpinPiece = false;
    }

    void attackClean(double& damage, StackLayer& topLayer) {
        if (topLayer.height < 4) {
            // attack with the layer below
            if (topLayer.open && hasI) {
                if (stack.size() > 1) {
                    StackLayer secondTopLayer = stack[stack.size() - 2];
                    attackClean(damage, secondTopLayer);
                }
                return;
            }
            topLayer.type = COMBO;
            attackCombo(damage, topLayer);
            return;
        }
        if (!hasI) {
            return;
        }
        damage = 4;
        reduceLayerHeight(topLayer, 4);

        hasI = false;
    }

    void attackMessy(double& damage, StackLayer& topLayer) {
        if (topLayer.height < 1) {
            stack.pop_back();
            return;
        }
        damage = 1;
        if (topLayer.height > 4) {
            reduceLayerHeight(topLayer, 0.55);
        }
        else {
            reduceLayerHeight(topLayer, 0.95);
        }
    }

    void attackCombo(double& damage, StackLayer& topLayer) {
        if (topLayer.height < 1) {
            stack.pop_back();
            return;
        }
        if (combo >= comboTable.size()) {
            damage = 5;
        }
        else {
            damage = comboTable[combo];
        }
        reduceLayerHeight(topLayer, 1);
        combo++;
    }

    double make_attack() {
        if (stack.empty()) return 0;

        StackLayer& topLayer = stack.back();

        double damage = 0;

        LayerType attackType = topLayer.type;

        double random_double = rng.getRand(100);
        bool failed_attack = false;
        failed_attack = failed_attack || ((attackType == SPIN) && (random_double < MISTAKE_PROB_SPIN));
        failed_attack = failed_attack || ((attackType == CLEAN) && (random_double < MISTAKE_PROB_CLEAN));
        failed_attack = failed_attack || ((attackType == COMBO) && (random_double < MISTAKE_PROB_COMBO));

        if (!can_attack()) {
            random_double = rng.getRand(100);
            if (random_double < MISTAKE_DOWNSTACK_PROB) {
                failed_attack = true;
            }
        }

        if (failed_attack) {
            attackType = MESSY;
        }

        switch (attackType) {
        case SPIN:
            attackSpin(damage, topLayer); break;
        case CLEAN:
            attackClean(damage, topLayer); break;
        case MESSY:
            attackMessy(damage, topLayer); break;
        case COMBO:
            attackCombo(damage, topLayer); break;
        default:
            break;
        }

        return damage;
    }

    void reduceLayerHeight(StackLayer& layer, double amount) {
        layer.height -= amount;
        if (layer.height <= 0) {
            stack.pop_back();
        }
    }

    void receiveAttack(double amount) {
        if (amount == 0) {
            return;
        }
        // Add garbage layers to the garbage meter
        if (amount < 3) {
            if (rng.getRand(100) < COMBO_GARBAGE_PROB) {
                garbage.push_back({ COMBO, (double)amount });
            }
            else {
                garbage.push_back({ MESSY, (double)amount });
            }
        }
        else {
            garbage.push_back({ CLEAN, (double) amount });
        }
    }

    double stack_height() {
        double height = 0;
        for (StackLayer& layer : stack) {
            if (layer.type == COMBO) {
                continue;
            }
            height += layer.height;
        }
        return height;
    }
    double garbage_height() {
        double height = 0;
        for (StackLayer& layer : garbage) {
            height += layer.height;
        }
        return height;
    }

    double process_meter(double attack) {

        double initial_attack = attack;

        // cancelling

        while (attack > 0 && !garbage.empty()) {
            double& incoming = garbage.back().height;

            if (incoming >= attack) {
                incoming -= attack;
                attack = 0;
            }
            else {
                attack -= incoming;
                incoming = 0;
            }

            if (incoming == 0) {
                garbage.pop_back();
            }
        }

        if (initial_attack > 0) {
            return attack;
        }

        // 8 garbage cap

        double garbage_used = 0;

        while (!garbage.empty()) {
            double& incoming = garbage.back().height;

            double garbage_amount = std::min(incoming, 8 - garbage_used);

            StackLayer& garbageLayer = garbage.back();
            stack.insert(stack.begin(), garbageLayer);

            incoming -= garbage_amount;
            garbage_used += garbage_amount;

            if (incoming == 0) {
                garbage.pop_back();
            }

            if (garbage_used == 8) {
                break;
            }
        }

        return attack;
    }

    void advance_queue() {
        if (nextI == 0) {
            hasI = true;
            nextI = rng.getRand(2) + 1;
        }
        if (nextSpinPiece == 0) {
            hasSpinPiece = true;
            nextSpinPiece = rng.getRand(2) + 1;
        }
        nextI--;
        nextSpinPiece--;
    }

    // Advances internal state and returns outgoing damage this turn
    double play() {

        if (dead) {
            // revive
            stack.clear();
            dead = false;
        }

        // disable death for noww
        if (stack_height() > 20) {
            dead = true;
            deaths++;
            return 0;
        }

        pieces++;

        nextState();

        double attack = 0;

        if (state == ATTACK && can_attack()) {
            attack = make_attack();
        }
        else {
            build();
        }

        // cancelling logic
        attack = process_meter((double) attack);

        advance_queue();

        if (rng.getRand(100) < WASTE_I_PROB) {
            hasI = false;
        }

        total_damage += attack;

        return attack;
    }
    
    bool is_dead() {
        return dead;
    }
};
