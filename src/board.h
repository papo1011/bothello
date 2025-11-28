#pragma once

#include <cstdint>
#include <ostream>
#include <tuple>
#include <utility>

typedef uint64_t Move;
typedef uint64_t MoveList;

// Assuming a board is 8x8
class Board {
  public:
    Board(uint64_t black_mask = 0, uint64_t white_mask = 0)
        : black_mask(black_mask)
        , white_mask(white_mask)
    {
    }

    // Flips the board each turn (ie swaps black_mask and white_mask)
    // Standard way of swapping two variables (faster than bitwise) if not optimised.
    // Use gcc -O2 or -O3 to fully optimise it.
    inline void flip()
    {
        uint64_t temp = this->black_mask;
        this->black_mask = this->white_mask;
        this->white_mask = temp;
    }

    // parameter `move` is an integer with a single bit set to 1 that represents where
    // the move is played. The function does not check if the move is legal.
    void move(Move move);

    // parameter `move` is an integer with a single bit set to 1 that represents where
    // the move is played.
    bool is_move_legal(Move move) const;

    MoveList list_available_legal_moves() const;

    bool is_there_a_legal_move_available() const;

    // Return { nb of black discs, nb of white discs }
    std::pair<unsigned, unsigned> score() const;

    inline void clear()
    {
        this->black_mask = 0;
        this->white_mask = 0;
    }

    inline uint64_t get_black_mask() const { return this->black_mask; }
    inline uint64_t get_white_mask() const { return this->white_mask; }

    friend std::ostream &operator<<(std::ostream &os, Board const &board);

  private:
    uint64_t black_mask; // Integer used as 8x8 grids
    uint64_t white_mask; // Integer used as 8x8 grids
};

std::ostream &operator<<(std::ostream &os, Board const &board);
