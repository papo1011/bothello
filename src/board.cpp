#include "board.h"
#include <cstdint>

// Converts the bitmask MoveList (uint64_t) into a generic vector of moves
std::vector<Move> get_moves_as_vector(MoveList move_list)
{
    std::vector<Move> moves;
    for (int i = 0; i < 64; ++i) {
        Move m = 1ULL << i;
        if (move_list & m) {
            moves.push_back(m);
        }
    }
    return moves;
}

// Converts a single Move (bitmask with one bit) into GTP string format
// GTP Go Text Protocol
std::string move_to_gtp(Move move)
{
    if (move == 0)
        return "pass";

    int idx = -1;
    for (int i = 0; i < 64; ++i) {
        if (move & (1ULL << i)) {
            idx = i;
            break;
        }
    }
    if (idx == -1)
        return "invalid";

    int row = idx / 8; // 0..7
    int col = idx % 8; // 0..7

    char col_char = 'a' + col; // a..h
    char row_char = '1' + row; // 1..8

    std::string res;
    res += col_char;
    res += row_char;
    return res;
}

std::ostream &operator<<(std::ostream &os, Board const &board)
{
    MoveList legal_moves = board.list_available_legal_moves();

    uint64_t black_pieces, white_pieces;

    if (board.is_black_turn()) {
        black_pieces = board.curr_player_mask;
        white_pieces = board.opp_player_mask;
    } else {
        white_pieces = board.curr_player_mask;
        black_pieces = board.opp_player_mask;
    }

    os << "  A B C D E F G H\n";
    for (int row = 0; row < 8; ++row) {
        os << (row + 1) << " ";
        for (int col = 0; col < 8; ++col) {
            int index = row * 8 + col;
            uint64_t mask = 1ULL << index;

            if (black_pieces & mask)
                os << "* ";
            else if (white_pieces & mask)
                os << "O ";
            else if (legal_moves & mask)
                os << ". "; // available move
            else
                os << "- ";
        }
        os << (row + 1) << "\n";
    }
    os << "  A B C D E F G H\n";
    return os;
}
