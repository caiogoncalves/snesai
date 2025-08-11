# SNES AI

This project uses Python to create AI agents that can play Super Nintendo (SNES) games, with a primary focus on Super Mario World. It includes a simple reactive agent and a more complex agent based on Deep Q-Learning (DQN).

## Features

*   **Reactive Agent**: A simple bot that uses template matching to identify and react to enemies on the screen.
*   **DQN Agent**: A reinforcement learning agent that learns to play the game through trial and error, using a deep neural network to make decisions.
*   **Game Visualization**: Scripts to visualize what the agent is "seeing" in real-time.
*   **Emulator Interaction**: Uses `pyautogui` to send keyboard commands to the SNES emulator (RetroArch) and `mss` to capture the screen.

## Project Structure

*   `run.py`: The main entry point to run the different agents and scripts.
*   `src/agents/dqn_agent.py`: The main script for training the DQN agent.
*   `src/agents/reactive_agent.py`: The script to run the template-based reactive agent.
*   `src/scripts/test_input.py`: A simple script to test game interaction.
*   `src/scripts/view_game.py`: A script to visualize the game screen as seen by the agent.
*   `assets/enemy_template.png`: The image template used by the reactive agent to detect enemies.
*   `runs/`: Directory where TensorBoard logs for DQN training are saved.

## Prerequisites

*   Python 3.x
*   A SNES emulator, such as [RetroArch](https://www.retroarch.com/)
*   The game's ROM (e.g., Super Mario World)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/snes-ai.git
    cd snes-ai
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

### 1. Configure the Emulator

1.  Open RetroArch and load the game's ROM.
2.  Ensure the game window is visible and not minimized.
3.  Adjust the screen coordinates in the script you want to run. The coordinates are in the `bounding_box` or `SCREEN_POS` variable.

### 2. Run the Scripts

Use the main `run.py` script to choose which component to execute:
```bash
python run.py
```
You will be prompted to choose one of the following:
*   Reactive Agent
*   DQN Agent
*   View Game Screen
*   Test Input

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
