// computes opponent::features for positions read from stdin, for scripts/train_encoder_hmm.py
// input, one position per line:
//   KEY BOARD CURRENT HOLD Q0 Q1 Q2 Q3 Q4 COMBO B2B METER
// BOARD is 400 hex digits: 20 rows of 10 cells, bottom row first, nonzero = filled. Pieces are one of
// IJLOSTZ, or N for none. Output: KEY and the features, space separated; the first line names the columns.

#include "OpponentFeatures.hpp"

#include <iostream>
#include <sstream>
#include <string>

static PieceType parsePiece(const std::string& text) {
    switch (text.empty() ? 'N' : text[0]) {
    case 'I': return PieceType::I;
    case 'J': return PieceType::J;
    case 'L': return PieceType::L;
    case 'O': return PieceType::O;
    case 'S': return PieceType::S;
    case 'T': return PieceType::T;
    case 'Z': return PieceType::Z;
    default: return PieceType::Empty;
    }
}

static int hexDigit(char c) {
    return c >= '0' && c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cout << "key";
    for (int f = 0; f < opponent::FEATURES; f++) {
        std::cout << ' ' << opponent::featureName(f);
    }
    std::cout << '\n';

    std::string line;
    long line_number = 0;
    while (std::getline(std::cin, line)) {
        line_number++;
        if (line.empty()) {
            continue;
        }
        std::istringstream in(line);
        std::string key, board, current, hold;
        std::string queue[opponent::PREVIEWS];
        opponent::View view;
        in >> key >> board >> current >> hold;
        for (auto& q : queue) {
            in >> q;
        }
        in >> view.combo >> view.b2b >> view.meter;
        if (!in || board.size() != 400) {
            std::cerr << "nana_opp_features: bad line " << line_number << std::endl;
            return 1;
        }
        for (int y = 0; y < 20; y++) {
            for (int x = 0; x < 10; x++) {
                const int i = 2 * (y * 10 + x);
                if (hexDigit(board[i]) * 16 + hexDigit(board[i + 1]) != 0) {
                    view.board.set(x, y);
                }
            }
        }
        view.current = parsePiece(current);
        const PieceType held = parsePiece(hold);
        if (held != PieceType::Empty) {
            view.hold = held;
        }
        for (int i = 0; i < opponent::PREVIEWS; i++) {
            view.queue[i] = parsePiece(queue[i]);
        }
        const auto f = opponent::features(view);
        std::cout << key;
        for (float x : f) {
            std::cout << ' ' << x;
        }
        std::cout << '\n';
    }
    return 0;
}
