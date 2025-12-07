#include "board.h"
#include <cstdint>

bool Board::is_move_legal(Move move) const
{
    if ((curr_player_mask | opp_player_mask) & move)
        return false;

    uint64_t opponent = opp_player_mask;
    uint64_t me = curr_player_mask;

    // Directions: E, W, S, N, SE, SW, NE, NW
    int shifts[] = {1, -1, 8, -8, 9, 7, -7, -9};

    uint64_t notA = 0xfefefefefefefefeULL;
    uint64_t notH = 0x7f7f7f7f7f7f7f7fULL;
    uint64_t all = 0xffffffffffffffffULL;

    uint64_t masks[] = {
        notH, notA, all,  all, // E, W, S, N
        notH, notA, notH, notA // SE, SW, NE, NW
    };

    for (int i = 0; i < 8; ++i) {
        int shift = shifts[i];
        uint64_t mask = masks[i];

        Move cursor = move;

        // First step
        if (!(cursor & mask))
            continue;
        if (shift > 0)
            cursor <<= shift;
        else
            cursor >>= -shift;

        if (!(cursor & opponent))
            continue; // Must be opponent

        // Scan
        while (true) {
            if (!(cursor & mask))
                break; // Hit edge
            if (shift > 0)
                cursor <<= shift;
            else
                cursor >>= -shift;

            if (cursor & me)
                return true; // Found bracket
            if (!(cursor & opponent))
                break; // Empty
        }
    }
    return false;
}

void Board::move(Move move)
{
    uint64_t opponent = opp_player_mask;
    uint64_t me = curr_player_mask;
    uint64_t flips = 0;

    // Directions: E, W, S, N, SE, SW, NE, NW
    int shifts[] = {1, -1, 8, -8, 9, 7, -7, -9};

    uint64_t notA = 0xfefefefefefefefeULL;
    uint64_t notH = 0x7f7f7f7f7f7f7f7fULL;
    uint64_t all = 0xffffffffffffffffULL;

    uint64_t masks[] = {
        notH, notA, all,  all, // E, W, S, N
        notH, notA, notH, notA // SE, SW, NE, NW
    };

    for (int i = 0; i < 8; ++i) {
        int shift = shifts[i];
        uint64_t mask = masks[i];

        Move cursor = move;
        uint64_t potential_flips = 0;

        // First step
        if (!(cursor & mask))
            continue;
        if (shift > 0)
            cursor <<= shift;
        else
            cursor >>= -shift;

        if (!(cursor & opponent))
            continue; // Must be opponent

        potential_flips |= cursor;

        // Scan
        while (true) {
            if (!(cursor & mask)) {
                potential_flips = 0; // Hit edge
                break;
            }
            if (shift > 0)
                cursor <<= shift;
            else
                cursor >>= -shift;

            if (cursor & me) {
                flips |= potential_flips; // Found bracket
                break;
            }
            if (!(cursor & opponent)) {
                potential_flips = 0; // Empty
                break;
            }
            potential_flips |= cursor;
        }
    }

    this->curr_player_mask |= move | flips;
    this->opp_player_mask &= ~flips;
    this->flip();
}

MoveList Board::list_available_legal_moves() const
{
    uint64_t legal_moves = 0;
    uint64_t empty = ~(curr_player_mask | opp_player_mask);

    // Iterate over all empty squares
    for (int i = 0; i < 64; ++i) {
        Move move = 1ULL << i;
        if (empty & move) {
            if (is_move_legal(move)) {
                legal_moves |= move;
            }
        }
    }
    return legal_moves;
}

bool Board::is_there_a_legal_move_available() const
{
    uint64_t empty = ~(curr_player_mask | opp_player_mask);

    for (int i = 0; i < 64; ++i) {
        Move move = 1ULL << i;
        if (empty & move) {
            if (is_move_legal(move)) {
                return true;
            }
        }
    }
    return false;
}

bool Board::is_win()
{
    if (!this->is_there_a_legal_move_available()) {
        this->flip();
        if (!this->is_there_a_legal_move_available()) {
            auto score = this->score();
            // We play as black but we flipped the board so we win if black < white
            return score.first < score.second;
        }
    }
    this->flip();
    return false;
}

bool Board::is_terminal()
{
    if (this->is_there_a_legal_move_available())
        return false;

    this->flip();
    bool opp_has_move = this->is_there_a_legal_move_available();
    this->flip();

    if (!opp_has_move)
        return true;
}

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

    if (board.is_black_turn) {
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
