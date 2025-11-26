#include "../src/board.h"
#include <gtest/gtest.h>

// Helper to create a bitmask from coordinates (row 0-7, col 0-7)
// Assumes row 0 is top, col 0 is left (A).
constexpr uint64_t BIT(int row, int col)
{
    return 1ULL << (row * 8 + col);
}

class BoardTest : public ::testing::Test {
protected:
    // Helper to visualize or setup if needed
};

TEST_F(BoardTest, EmptyBoardNoMoves)
{
    // If board is empty, no move is legal because you can't flank anything
    Board board(0, 0);
    // Try all positions
    for (int i = 0; i < 64; ++i) {
        EXPECT_FALSE(board.is_move_legal(1ULL << i));
    }
}

TEST_F(BoardTest, SimpleHorizontal)
{
    // Black at A1 (0,0), White at B1 (0,1).
    // Legal move for Black should be C1 (0,2).
    uint64_t black = BIT(0, 0);
    uint64_t white = BIT(0, 1);
    Board board(black, white);

    EXPECT_TRUE(board.is_move_legal(BIT(0, 2)));

    // D1 should not be legal (gap)
    EXPECT_FALSE(board.is_move_legal(BIT(0, 3)));
}

TEST_F(BoardTest, SimpleVertical)
{
    // Black at A1 (0,0), White at A2 (1,0).
    // Legal move for Black should be A3 (2,0).
    uint64_t black = BIT(0, 0);
    uint64_t white = BIT(1, 0);
    Board board(black, white);

    EXPECT_TRUE(board.is_move_legal(BIT(2, 0)));
}

TEST_F(BoardTest, SimpleDiagonal)
{
    // Black at A1 (0,0), White at B2 (1,1).
    // Legal move for Black should be C3 (2,2).
    uint64_t black = BIT(0, 0);
    uint64_t white = BIT(1, 1);
    Board board(black, white);

    EXPECT_TRUE(board.is_move_legal(BIT(2, 2)));
}

TEST_F(BoardTest, ReverseDirection)
{
    // Black at C1 (0,2), White at B1 (0,1).
    // Legal move for Black should be A1 (0,0).
    uint64_t black = BIT(0, 2);
    uint64_t white = BIT(0, 1);
    Board board(black, white);

    EXPECT_TRUE(board.is_move_legal(BIT(0, 0)));
}

TEST_F(BoardTest, OccupiedSquare)
{
    // Black at A1, White at B1.
    // C1 is legal.
    // But if C1 is occupied by White, it's not legal (it's occupied).
    // If C1 is occupied by Black, it's not legal.

    uint64_t black = BIT(0, 0);
    uint64_t white = BIT(0, 1);

    // Case 1: Target occupied by Black
    Board board1(black | BIT(0, 2), white);
    EXPECT_FALSE(board1.is_move_legal(BIT(0, 2)));

    // Case 2: Target occupied by White
    Board board2(black, white | BIT(0, 2));
    EXPECT_FALSE(board2.is_move_legal(BIT(0, 2)));
}

TEST_F(BoardTest, EdgeWrappingHorizontal)
{
    // H1 (0,7) is Black. A2 (1,0) is White.
    // They are adjacent in memory (bits 7 and 8).
    // But they are not adjacent on the board.
    // Move to B2 (1,1) should NOT be legal if it relies on wrapping.

    uint64_t black = BIT(0, 7);
    uint64_t white = BIT(1, 0);
    Board board(black, white);

    EXPECT_FALSE(board.is_move_legal(BIT(1, 1)));
}

TEST_F(BoardTest, StandardOpening)
{
    // Standard Othello starting position:
    // White: D4, E5
    // Black: E4, D5
    // Coordinates:
    // D4 = (3, 3), E4 = (3, 4)
    // D5 = (4, 3), E5 = (4, 4)

    uint64_t white = BIT(3, 3) | BIT(4, 4);
    uint64_t black = BIT(3, 4) | BIT(4, 3);

    Board board(black, white);

    // Legal moves for Black:
    // D3 (2, 3) -> Flanks D4 (White) with D5 (Black) - Vertical
    EXPECT_TRUE(board.is_move_legal(BIT(2, 3)));

    // C4 (3, 2) -> Flanks D4 (White) with E4 (Black) - Horizontal
    EXPECT_TRUE(board.is_move_legal(BIT(3, 2)));

    // F5 (4, 5) -> Flanks E5 (White) with D5 (Black) - Horizontal
    EXPECT_TRUE(board.is_move_legal(BIT(4, 5)));

    // E6 (5, 4) -> Flanks E5 (White) with E4 (Black) - Vertical
    EXPECT_TRUE(board.is_move_legal(BIT(5, 4)));

    // Illegal moves:
    // E3 (2, 4) -> No flank. (E4 is Black, E5 is White... wait)
    // E3 (2, 4). Neighbor E4 is Black. No capture.
    EXPECT_FALSE(board.is_move_legal(BIT(2, 4)));
}
