#pragma once

#include <cstdint>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

// Enable GPU compatibility when compiled with CUDA
#ifdef __CUDACC__
#    define HOST_DEVICE __host__ __device__
#else
#    define HOST_DEVICE
#endif

typedef uint64_t Move;
typedef uint64_t MoveList;

// =============================================================================
// GPU-Compatible Board Implementation (DeviceBoard)
// =============================================================================
// Lightweight board representation optimized for GPU kernel execution.
// Can also be used on CPU thanks to HOST_DEVICE macros.
// Uses bitboard representation identical to Board but with GPU-optimized methods.
// 8x8 board represented as two 64-bit masks
class Board {
  public:
    HOST_DEVICE Board(uint64_t curr_player_mask = 0, uint64_t opp_player_mask = 0,
                      bool is_black_to_move = true)
        : curr_player_mask(curr_player_mask)
        , opp_player_mask(opp_player_mask)
        , is_black_to_move(is_black_to_move)
    {
    }

    // Flips the board each turn (ie swaps curr_player_mask and opp_player_mask)
    // Standard way of swapping two variables (faster than bitwise) if not optimised.
    // Use gcc -O2 or -O3 to fully optimise it.
    HOST_DEVICE inline void flip()
    {
        uint64_t temp = this->curr_player_mask;
        this->curr_player_mask = this->opp_player_mask;
        this->opp_player_mask = temp;

        this->is_black_to_move = !this->is_black_to_move;
    }

    // parameter `move` is an integer with a single bit set to 1 that represents where
    // the move is played. The function does not check if the move is legal.
    HOST_DEVICE void move(Move move)
    {
        if (move == 0) {
            this->flip();
            return;
        }

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

    // parameter `move` is an integer with a single bit set to 1 that represents where
    // the move is played.
    HOST_DEVICE bool is_move_legal(Move move) const
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

    HOST_DEVICE MoveList list_available_legal_moves() const
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

    HOST_DEVICE bool is_there_a_legal_move_available() const
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

    HOST_DEVICE bool is_win()
    {
        if (!this->is_there_a_legal_move_available()) {
            this->flip();
            if (!this->is_there_a_legal_move_available()) {
                int my_score, opp_score;
                this->get_score(my_score, opp_score);
                // We play as black but we flipped the board so we win if black < white
                return my_score < opp_score;
            }
        }
        this->flip();
        return false;
    }

    HOST_DEVICE bool is_terminal()
    {
        if (this->is_there_a_legal_move_available())
            return false;

        this->flip();
        bool opp_has_move = this->is_there_a_legal_move_available();
        this->flip();

        if (!opp_has_move)
            return true;
        return false;
    }

    // Returns score for (current_player, opponent)
    // Numbers of black disks / number of white disks
    HOST_DEVICE void get_score(int &my_score, int &opp_score) const
    {
#ifdef __CUDA_ARCH__
        my_score = __popcll(curr_player_mask);
        opp_score = __popcll(opp_player_mask);
#else
        my_score = __builtin_popcountll(curr_player_mask);
        opp_score = __builtin_popcountll(opp_player_mask);
#endif
    }

    HOST_DEVICE static int count_moves(MoveList moves)
    {
#ifdef __CUDA_ARCH__
        return __popcll(moves);
#else
        return __builtin_popcountll(moves);
#endif
    }

    // Resets the board to empty.
    HOST_DEVICE inline void clear()
    {
        this->curr_player_mask = 0;
        this->opp_player_mask = 0;
    }

    HOST_DEVICE inline uint64_t get_curr_player_mask() const
    {
        return this->curr_player_mask;
    }
    HOST_DEVICE inline uint64_t get_opp_player_mask() const
    {
        return this->opp_player_mask;
    }

    // Helper to get the k-th set bit (0-indexed)
    HOST_DEVICE static Move get_nth_move(MoveList moves, int n)
    {
        // Iterate to find n-th bit
        int count = 0;
        for (int i = 0; i < 64; ++i) {
            Move m = 1ULL << i;
            if (moves & m) {
                if (count == n)
                    return m;
                count++;
            }
        }
        return 0;
    }

    HOST_DEVICE inline bool is_black_turn() const { return this->is_black_to_move; }

    friend std::ostream &operator<<(std::ostream &os, Board const &board);

  private:
    uint64_t curr_player_mask; // Integer used as 8x8 grids.
    uint64_t opp_player_mask;  // Integer used as 8x8 grids.
    bool is_black_to_move;
};

std::vector<Move> get_moves_as_vector(MoveList move_list);
std::string move_to_gtp(Move move);
std::ostream &operator<<(std::ostream &os, Board const &board);
