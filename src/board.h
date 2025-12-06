#pragma once

#include <cstdint>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

typedef uint64_t Move;
typedef uint64_t MoveList;

// 8x8 board represented as two 64-bit masks
class Board {
  public:
    Board(uint64_t curr_player_mask = 0, uint64_t opp_player_mask = 0,
          bool is_black_turn = true)
        : curr_player_mask(curr_player_mask)
        , opp_player_mask(opp_player_mask)
        , is_black_turn(is_black_turn)
    {
    }

    // Flips the board each turn (ie swaps curr_player_mask and opp_player_mask)
    // Standard way of swapping two variables (faster than bitwise) if not optimised.
    // Use gcc -O2 or -O3 to fully optimise it.
    inline void flip()
    {
        uint64_t temp = this->curr_player_mask;
        this->curr_player_mask = this->opp_player_mask;
        this->opp_player_mask = temp;

        this->is_black_turn = !this->is_black_turn;
    }

    // parameter `move` is an integer with a single bit set to 1 that represents where
    // the move is played. The function does not check if the move is legal.
    void move(Move move);

    // parameter `move` is an integer with a single bit set to 1 that represents where
    // the move is played.
    bool is_move_legal(Move move) const;

    MoveList list_available_legal_moves() const;

    bool is_there_a_legal_move_available() const;

    bool is_win();

    inline bool is_terminal()
    {
        // The order of the OR is important here.
        return this->is_win() || this->is_score_draw();
    }

    // Return { nb of black discs, nb of white discs }.
    inline std::pair<unsigned, unsigned> score() const
    {
        return {__builtin_popcountll(this->curr_player_mask),
                __builtin_popcountll(this->opp_player_mask)};
    }

    inline bool is_score_draw() const
    {
        auto score = this->score();
        return score.first == score.second;
    }

    // Resets the board to empty.
    inline void clear()
    {
        this->curr_player_mask = 0;
        this->opp_player_mask = 0;
    }

    inline uint64_t get_curr_player_mask() const { return this->curr_player_mask; }
    inline uint64_t get_opp_player_mask() const { return this->opp_player_mask; }

    friend std::ostream &operator<<(std::ostream &os, Board const &board);

  private:
    uint64_t curr_player_mask; // Integer used as 8x8 grids.
    uint64_t opp_player_mask;  // Integer used as 8x8 grids.
    bool is_black_turn;
};

std::vector<Move> get_moves_as_vector(MoveList move_list);
std::string move_to_gtp(Move move);
std::ostream &operator<<(std::ostream &os, Board const &board);
