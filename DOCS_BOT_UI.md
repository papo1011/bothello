# Play vs Bot (UI)

This repository contains a small frontend in `full_stack-othello_game` that can run locally and play against the bots compiled in this repo.

The UI was done by Fabien CHAVONET: https://github.com/fchavonet/full_stack-othello_game

Quick steps:

1. Build the C++ project (ensure `bot_cli` is created):

   mkdir -p build && cd build && cmake .. && make -j

2. Install Node dependencies and run the server:

   npm install
   npm start

3. Open http://localhost:3000 in your browser. In the header select `Single Player`, pick a bot and a time (ms), then play against it.

Notes:
- The server assumes the `bot_cli` executable is at `./build/bot_cli` relative to the repository root.
- GPU-backed bots (leaf/block/cuda) require a GPU and a CUDA-capable build; if unavailable they may fail at execution time.

Persistent bot processes:
- The server can keep a bot process in memory across multiple moves (reduces GPU reinitialization and ensures the same bot instance serves the whole game).
- The UI will receive a `session` identifier in the bot response; that `session` is automatically stored and reused by the frontend so subsequent moves are handled by the same bot process.

Player color:
 - You can choose `Play as: Black` or `Play as: White` in the header. If you select `White`, the bot will play Black and move first.
 - While in single-player mode you can only play the color you selected; clicking is disabled when it's the bot's turn so you cannot play both colors.
