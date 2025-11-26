#include "src/board.h"
#include <iostream>
#include <unistd.h>

int main()
{
    Board board;
    board.move(1);
    board.move(3);
    std::cout << board;
    return 0;
}
