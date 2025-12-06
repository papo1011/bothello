#include "../src/board.h"
#include <gtest/gtest.h>
#include <ostream>

// Helper to create a bitmask from coordinates (row 0-7, col 0-7)
// Assumes row 0 is top, col 0 is left (A).
constexpr uint64_t BIT(int row, int col) { return 1ULL << (row * 8 + col); }

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

TEST_F(BoardTest, MoveFlipsDiscsHorizontal)
{
    // Black at A1 (0,0), White at B1 (0,1).
    // Move at C1 (0,2).
    uint64_t black = BIT(0, 0);
    uint64_t white = BIT(0, 1);
    Board board(black, white);

    board.move(BIT(0, 2));

    // After move, it's White's turn.
    // So board.curr_player_mask is White's pieces (should be 0)
    // board.opp_player_mask is Black's pieces (should be A1 | B1 | C1)

    EXPECT_EQ(board.get_curr_player_mask(), 0ULL);
    EXPECT_EQ(board.get_opp_player_mask(), BIT(0, 0) | BIT(0, 1) | BIT(0, 2));
}

TEST_F(BoardTest, MoveFlipsDiscsVertical)
{
    // Black at A1 (0,0), White at A2 (1,0).
    // Move at A3 (2,0).
    uint64_t black = BIT(0, 0);
    uint64_t white = BIT(1, 0);
    Board board(black, white);

    board.move(BIT(2, 0));

    // After move, it's White's turn.
    // So board.curr_player_mask is White's pieces (should be 0)
    // board.opp_player_mask is Black's pieces (should be A1 | A2 | A3)

    EXPECT_EQ(board.get_curr_player_mask(), 0ULL);
    EXPECT_EQ(board.get_opp_player_mask(), BIT(0, 0) | BIT(1, 0) | BIT(2, 0));
}

TEST_F(BoardTest, MoveFlipsDiscsDiagonal)
{
    // Black at A1 (0,0), White at B2 (1,1).
    // Move at C3 (2,2).
    uint64_t black = BIT(0, 0);
    uint64_t white = BIT(1, 1);
    Board board(black, white);

    board.move(BIT(2, 2));

    // After move, it's White's turn.
    // So board.curr_player_mask is White's pieces (should be 0)
    // board.opp_player_mask is Black's pieces (should be A1 | B2 | C3)

    EXPECT_EQ(board.get_curr_player_mask(), 0ULL);
    EXPECT_EQ(board.get_opp_player_mask(), BIT(0, 0) | BIT(1, 1) | BIT(2, 2));
}

TEST_F(BoardTest, MoveFlipsMultipleDirections)
{
    // Setup a scenario where one move flips in 3 directions (Horizontal, Vertical,
    // Diagonal) Black pieces at: A1(0,0), C1(0,2), A3(2,0) White pieces at: A2(1,0),
    // B2(1,1), B1(0,1) Move at C3(2,2) should flip B2 (Diagonal) -> No, wait.

    // Let's construct it carefully.
    // Center of action: B2 (1,1) - This is where we play? No, we play to flank.

    // Let's play at B2 (1,1).
    // We need Black pieces surrounding it, and White pieces in between.

    // Direction West (Left): A2(1,0) is White, A1(1,-1 invalid) -> Let's shift.
    // Play at C3 (2,2).
    // 1. North (Up): C2(1,2) White, C1(0,2) Black. -> Flips C2.
    // 2. West (Left): B3(2,1) White, A3(2,0) Black. -> Flips B3.
    // 3. North-West (Diag): B2(1,1) White, A1(0,0) Black. -> Flips B2.

    uint64_t black = BIT(0, 2) | BIT(2, 0) | BIT(0, 0);
    uint64_t white = BIT(1, 2) | BIT(2, 1) | BIT(1, 1);

    Board board(black, white);

    // std::cout << board << std::endl;

    // Verify move is legal
    EXPECT_TRUE(board.is_move_legal(BIT(2, 2)));

    board.move(BIT(2, 2));

    // std::cout << board << std::endl;

    // After move, it's White's turn.
    // board.opp_player_mask should contain all the pieces now (original black +
    // original white + new piece) because all white pieces were flipped to black.

    uint64_t expected_black_pieces = black | white | BIT(2, 2);

    EXPECT_EQ(board.get_curr_player_mask(), 0ULL);
    EXPECT_EQ(board.get_opp_player_mask(), expected_black_pieces);
}

TEST_F(BoardTest, StandardOpeningMove)
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

    // Black plays C4 (3, 2).
    // This should flip D4 (3, 3) which is White.
    // Result: D4 becomes Black.

    board.move(BIT(3, 2));

    // Now it's White's turn.
    // board.curr_player_mask is White's pieces.
    // board.opp_player_mask is Black's pieces.

    // White pieces: E5 (4, 4) (unchanged)
    // Black pieces: E4 (3, 4), D5 (4, 3), C4 (3, 2) (new), D4 (3, 3) (flipped)

    EXPECT_EQ(board.get_curr_player_mask(), BIT(4, 4));
    EXPECT_EQ(board.get_opp_player_mask(),
              BIT(3, 4) | BIT(4, 3) | BIT(3, 2) | BIT(3, 3));
}

TEST_F(BoardTest, ListAvailableMoves)
{
    // Standard Othello starting position:
    // White: D4, E5
    // Black: E4, D5
    uint64_t white = BIT(3, 3) | BIT(4, 4);
    uint64_t black = BIT(3, 4) | BIT(4, 3);

    Board board(black, white);

    // Legal moves for Black:
    // D3 (2, 3), C4 (3, 2), F5 (4, 5), E6 (5, 4)
    uint64_t expected_moves = BIT(2, 3) | BIT(3, 2) | BIT(4, 5) | BIT(5, 4);

    EXPECT_EQ(board.list_available_legal_moves(), expected_moves);
    EXPECT_TRUE(board.is_there_a_legal_move_available());
}

TEST_F(BoardTest, NoAvailableMoves)
{
    Board board(0, 0);
    EXPECT_EQ(board.list_available_legal_moves(), 0ULL);
    EXPECT_FALSE(board.is_there_a_legal_move_available());
}

TEST_F(BoardTest, Score)
{
    // Black: 2 pieces, White: 2 pieces
    uint64_t black = BIT(3, 4) | BIT(4, 3);
    uint64_t white = BIT(3, 3) | BIT(4, 4);
    Board board(black, white);

    std::pair<uint64_t, uint64_t> s = board.score();
    EXPECT_EQ(s.first, 2ULL);
    EXPECT_EQ(s.second, 2ULL);

    // Make a move
    // Black plays C4 (3, 2). Flips D4 (3, 3).
    board.move(BIT(3, 2));

    // Black: E4, D5, C4, D4 (4 pieces)
    // White: E5 (1 piece)
    s = board.score();
    // Note: board.score() returns {curr_player_mask count, opp_player_mask count}
    // But after move, masks are swapped?
    // Board::move() calls flip().
    // So board.curr_player_mask is now the NEXT player (White).
    // board.opp_player_mask is the PREVIOUS player (Black).

    // Let's check the implementation of score().
    // return { __builtin_popcountll(curr_player_mask),
    // __builtin_popcountll(opp_player_mask)
    // };

    // So s.first is current curr_player_mask (White's pieces), s.second is current
    // opp_player_mask (Black's pieces).

    EXPECT_EQ(s.first, 1ULL);  // White has 1 piece
    EXPECT_EQ(s.second, 4ULL); // Black has 4 pieces
}
