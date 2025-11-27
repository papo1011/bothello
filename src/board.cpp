#include "board.h"
#include <cstdint>

bool Board::is_move_legal(uint64_t move)
{
    if ((black_mask | white_mask) & move)
        return false;

    uint64_t opponent = white_mask;
    uint64_t me = black_mask;

    // Directions: E, W, S, N, SE, SW, NE, NW
    int shifts[] = { 1, -1, 8, -8, 9, 7, -7, -9 };

    uint64_t notA = 0xfefefefefefefefeULL;
    uint64_t notH = 0x7f7f7f7f7f7f7f7fULL;
    uint64_t all = 0xffffffffffffffffULL;

    uint64_t masks[] = {
        notH, notA, all, all,  // E, W, S, N
        notH, notA, notH, notA // SE, SW, NE, NW
    };

    for (int i = 0; i < 8; ++i) {
        int shift = shifts[i];
        uint64_t mask = masks[i];

        uint64_t cursor = move;

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

void Board::move(uint64_t move)
{
    uint64_t opponent = white_mask;
    uint64_t me = black_mask;
    uint64_t flips = 0;

    // Directions: E, W, S, N, SE, SW, NE, NW
    int shifts[] = { 1, -1, 8, -8, 9, 7, -7, -9 };

    uint64_t notA = 0xfefefefefefefefeULL;
    uint64_t notH = 0x7f7f7f7f7f7f7f7fULL;
    uint64_t all = 0xffffffffffffffffULL;

    uint64_t masks[] = {
        notH, notA, all, all,  // E, W, S, N
        notH, notA, notH, notA // SE, SW, NE, NW
    };

    for (int i = 0; i < 8; ++i) {
        int shift = shifts[i];
        uint64_t mask = masks[i];

        uint64_t cursor = move;
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

    this->black_mask |= move | flips;
    this->white_mask &= ~flips;
    this->flip();
}

std::ostream& operator<<(std::ostream& os, Board const& board)
{
    os << "  A B C D E F G H\n";
    for (int row = 0; row < 8; ++row) {
        os << (row + 1) << " ";
        for (int col = 0; col < 8; ++col) {
            int index = row * 8 + col;
            uint64_t mask = 1ULL << index;

            if (board.black_mask & mask)
                os << "X ";
            else if (board.white_mask & mask)
                os << "O ";
            else
                os << ". ";
        }
        os << (row + 1) << "\n";
    }
    os << "  A B C D E F G H\n";
    return os;
}
