#pragma once

#include <cstdint>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

// Enable GPU compatibility when compiled with CUDA
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE 
#endif

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

    bool is_terminal();

    // Return { nb of black discs, nb of white discs }.
    inline std::pair<unsigned, unsigned> score() const
    {
        return {__builtin_popcountll(this->curr_player_mask),
                __builtin_popcountll(this->opp_player_mask)};
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

// =============================================================================
// GPU-Compatible Board Implementation (DeviceBoard)
// =============================================================================
// Lightweight board representation optimized for GPU kernel execution.
// Can also be used on CPU thanks to HOST_DEVICE macros.
// Uses bitboard representation identical to Board but with GPU-optimized methods.

namespace gpu {

// Type aliases for GPU namespace (same as global types)
typedef uint64_t Move;
typedef uint64_t MoveList;

struct DeviceBoard {
    uint64_t curr_player_mask;
    uint64_t opp_player_mask;
    bool is_black_turn;

    HOST_DEVICE DeviceBoard() 
        : curr_player_mask(0), opp_player_mask(0), is_black_turn(true) {}

    HOST_DEVICE DeviceBoard(uint64_t curr, uint64_t opp, bool black_turn)
        : curr_player_mask(curr), opp_player_mask(opp), is_black_turn(black_turn) {}

    HOST_DEVICE void flip() {
        uint64_t temp = curr_player_mask;
        curr_player_mask = opp_player_mask;
        opp_player_mask = temp;
        is_black_turn = !is_black_turn;
    }

    HOST_DEVICE void move(Move move) {
        uint64_t opponent = opp_player_mask;
        uint64_t me = curr_player_mask;
        uint64_t flips = 0;

        // E, W, S, N, SE, SW, NE, NW
        const int shifts[] = {1, -1, 8, -8, 9, 7, -7, -9};
        
        const uint64_t notA = 0xfefefefefefefefeULL;
        const uint64_t notH = 0x7f7f7f7f7f7f7f7fULL;
        const uint64_t all = 0xffffffffffffffffULL;

        const uint64_t masks_table[] = {
            notH, notA, all,  all, // E, W, S, N
            notH, notA, notH, notA // SE, SW, NE, NW
        };

        for (int i = 0; i < 8; ++i) {
            int shift = shifts[i];
            uint64_t mask = masks_table[i];

            Move cursor = move;
            uint64_t potential_flips = 0;

            if (!(cursor & mask)) continue;
            
            if (shift > 0) cursor <<= shift;
            else cursor >>= -shift;

            if (!(cursor & opponent)) continue;

            potential_flips |= cursor;

            while (true) {
                if (!(cursor & mask)) {
                    potential_flips = 0;
                    break;
                }
                if (shift > 0) cursor <<= shift;
                else cursor >>= -shift;

                if (cursor & me) {
                    flips |= potential_flips;
                    break;
                }
                if (!(cursor & opponent)) {
                    potential_flips = 0;
                    break;
                }
                potential_flips |= cursor;
            }
        }

        curr_player_mask |= move | flips;
        opp_player_mask &= ~flips;
        flip();
    }

    HOST_DEVICE bool is_move_legal(Move move) const {
        if ((curr_player_mask | opp_player_mask) & move) return false;

        uint64_t opponent = opp_player_mask;
        uint64_t me = curr_player_mask;
        
        const int shifts[] = {1, -1, 8, -8, 9, 7, -7, -9};
        const uint64_t notA = 0xfefefefefefefefeULL;
        const uint64_t notH = 0x7f7f7f7f7f7f7f7fULL;
        const uint64_t all = 0xffffffffffffffffULL;
        const uint64_t masks_table[] = {
            notH, notA, all,  all, 
            notH, notA, notH, notA 
        };

        for (int i = 0; i < 8; ++i) {
            int shift = shifts[i];
            uint64_t mask = masks_table[i];
            Move cursor = move;

            if (!(cursor & mask)) continue;

            if (shift > 0) cursor <<= shift;
            else cursor >>= -shift;

            if (!(cursor & opponent)) continue;

            while (true) {
                if (!(cursor & mask)) break;
                if (shift > 0) cursor <<= shift;
                else cursor >>= -shift;

                if (cursor & me) return true;
                if (!(cursor & opponent)) break;
            }
        }
        return false;
    }

    HOST_DEVICE MoveList list_available_legal_moves() const {
        uint64_t legal_moves = 0;
        uint64_t empty = ~(curr_player_mask | opp_player_mask);
        
        // Iterate over all 64 positions to check legality
        for (int i = 0; i < 64; ++i) {
            Move m = 1ULL << i;
            if (empty & m) {
                if (is_move_legal(m)) {
                    legal_moves |= m;
                }
            }
        }
        return legal_moves;
    }

    // Returns score for (current_player, opponent)
    HOST_DEVICE void get_score(int& my_score, int& opp_score) const {
        #ifdef __CUDA_ARCH__
        my_score = __popcll(curr_player_mask);
        opp_score = __popcll(opp_player_mask);
        #else
        my_score = __builtin_popcountll(curr_player_mask);
        opp_score = __builtin_popcountll(opp_player_mask);
        #endif
    }
    
    // Helper to get the k-th set bit (0-indexed)
    HOST_DEVICE static Move get_nth_move(MoveList moves, int n) {
        // Iterate to find n-th bit
        int count = 0;
        for(int i=0; i<64; ++i) {
            Move m = 1ULL << i;
            if (moves & m) {
                if (count == n) return m;
                count++;
            }
        }
        return 0;
    }
    
    HOST_DEVICE static int count_moves(MoveList moves) {
        #ifdef __CUDA_ARCH__
        return __popcll(moves);
        #else
        return __builtin_popcountll(moves);
        #endif
    }
};

} // namespace gpu
