"Deep Q-Learning (DQN) agent for playing SNES games.\n\nThis script implements a DQN agent that learns to play SNES games by\ninteracting with the emulator, capturing screen frames, and making decisions\nbased on a deep neural network. The agent's configuration and hyperparameters\nare loaded from the `src/config.py` file.\n\n"

import torch
import numpy as np
import cv2
from mss import mss
import pyautogui
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os
import sys

# Add the 'src' directory to the path to import the config module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    ACTIONS,
    ACTION_SPACE_SIZE,
    SCREEN_POS,
    FRAME_STACK_SIZE,
    INPUT_SHAPE,
    GAMMA,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    LEARNING_RATE,
    MEMORY_SIZE,
    BATCH_SIZE,
    TARGET_UPDATE_FREQUENCY
)

from dqn_agent_class import DQNAgent
from dqn_model import DQN

<<<<<<< HEAD

def process_frame(frame):
    """
    Converts a frame to grayscale and resizes it.

    Args:
        frame (numpy.ndarray): The raw frame captured from the screen.

    Returns:
        numpy.ndarray: The processed frame in grayscale and resized to 84x84.
    """
=======
# --- SETTINGS AND HYPERPARAMETERS ---

# UPDATED ACTIONS TO HANDLE RAMPS
ACTIONS = {
    0: 'right',             # Walk right
    1: ['right', 'x'],      # Run right (affects jumps)
    2: 'z',                 # Jump in place (short jump)
    3: ['right', 'z'],      # Jump right (normal jump)
    4: ['right', 'x', 'z'], # Run and jump right (long jump)
    5: ['up', 'right'],     # Crucial action for climbing ramps!
}
ACTION_SPACE_SIZE = len(ACTIONS)

# Screen settings
SCREEN_POS = {'top': 100, 'left': 100, 'width': 800, 'height': 600} # ADJUST FOR YOUR SCREEN
FRAME_STACK_SIZE = 4
INPUT_SHAPE = (FRAME_STACK_SIZE, 84, 84)

# DQN Hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
LEARNING_RATE = 0.00025
MEMORY_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 10

# --- IMAGE PROCESSING FUNCTION ---

def process_frame(frame):
    """Converts a frame to grayscale and resizes it."""
>>>>>>> a54129cf58c1ab5fe6d5ef1f3c03ba1870c41dda
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)

<<<<<<< HEAD
=======
# --- MAIN LOOP ---
>>>>>>> a54129cf58c1ab5fe6d5ef1f3c03ba1870c41dda

def run_dqn_training():
    """
    Runs the main training loop for the DQN agent.
    """
    agent = DQNAgent(
        input_shape=INPUT_SHAPE,
        num_actions=ACTION_SPACE_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        memory_size=MEMORY_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY
    )
    sct = mss()
    writer = SummaryWriter(f"runs/mario_dqn_{int(time.time())}")

    try:
        # Attempt to load benchmark images for Q-value analysis
        benchmark_images = {
<<<<<<< HEAD
            "Start": process_frame(cv2.imread("benchmark_inicio.png")),
            "Enemy": process_frame(cv2.imread("benchmark_inimigo.png"))
=======
            "Start": process_frame(cv2.imread("benchmark_start.png")),
            "Enemy": process_frame(cv2.imread("benchmark_enemy.png"))
>>>>>>> a54129cf58c1ab5fe6d5ef1f3c03ba1870c41dda
        }
        print("Benchmark images loaded successfully.")
    except Exception as e:
        print(f"WARNING: Could not load benchmark images. Error: {e}")
        benchmark_images = None

    # Initial frame capture and state stack setup
    frame = sct.grab(SCREEN_POS)
    processed_frame = process_frame(np.array(frame))
    state_stack = deque([processed_frame] * FRAME_STACK_SIZE, maxlen=FRAME_STACK_SIZE)
    current_state = np.array(state_stack)

    episode = 0
    total_steps = 0

    print("Starting training in 5 seconds. Click on the game window!")
    time.sleep(5)

    while True:
        episode += 1
        episode_reward = 0
        episode_steps = 0

        episode_losses = []
        action_counts = {i: 0 for i in range(ACTION_SPACE_SIZE)}

        print(f"\n--- Episode {episode} | Epsilon: {agent.epsilon:.4f} ---")
<<<<<<< HEAD
        # IMPORTANT: You need to manually reset the game here or automate it
=======
        # HERE YOU SHOULD RESET THE GAME MANUALLY OR AUTOMATE IT
>>>>>>> a54129cf58c1ab5fe6d5ef1f3c03ba1870c41dda
        time.sleep(2)

        done = False
        while not done:
            action_idx = agent.act(current_state)
            action_keys = ACTIONS[action_idx]

            action_counts[action_idx] += 1

            if isinstance(action_keys, list):
                for key in action_keys:
                    pyautogui.keyDown(key)
                time.sleep(0.05)
                for key in action_keys:
                    pyautogui.keyUp(key)
            else:
                pyautogui.press(action_keys)

<<<<<<< HEAD
            # Reward logic (NEEDS IMPROVEMENT)
=======
            # Reward logic (STILL NEEDS IMPROVEMENT)
>>>>>>> a54129cf58c1ab5fe6d5ef1f3c03ba1870c41dda
            reward = 0.1
            if episode_steps > 500:
                done = True
                reward = -10

            next_frame_raw = sct.grab(SCREEN_POS)
            processed_next_frame = process_frame(np.array(next_frame_raw))
            state_stack.append(processed_next_frame)
            next_state = np.array(state_stack)

            agent.remember(current_state, action_idx, reward, next_state, done)
            current_state = next_state

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            loss = agent.replay(BATCH_SIZE)

            if loss is not None:
                episode_losses.append(loss)
                writer.add_scalar('Training/Step_Loss', loss, total_steps)

<<<<<<< HEAD
        # End of episode report
        print("\n" + "=" * 30)
        print(f"EPISODE {episode} REPORT")
        print("=" * 30)
=======
        # Print end-of-episode report
        print("\n" + "="*30)
        print(f"EPISODE {episode} REPORT")
        print("="*30)
>>>>>>> a54129cf58c1ab5fe6d5ef1f3c03ba1870c41dda
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        print(f"  Performance:\n    - Total Reward: {episode_reward:.2f}\n    - Duration: {episode_steps} steps")
        print(f"  Training:\n    - Average Loss: {avg_loss:.5f}")
        total_actions = sum(action_counts.values())
        print(f"  Behavior (Action Distribution):")
        for idx, count in action_counts.items():
            action_name = " ".join(ACTIONS[idx]) if isinstance(ACTIONS[idx], list) else ACTIONS[idx]
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            print(f"    - Action '{action_name}': {count} times ({percentage:.1f}%)")
        if benchmark_images:
<<<<<<< HEAD
            print(f"  Q-Value Analysis ('Network Confidence'):")
=======
            print(f"  Q-Values Analysis (Network 'Confidence'):")
>>>>>>> a54129cf58c1ab5fe6d5ef1f3c03ba1870c41dda
            agent.policy_net.eval()
            with torch.no_grad():
                for name, img in benchmark_images.items():
                    bench_state = np.array([img] * FRAME_STACK_SIZE)
                    state_tensor = torch.tensor(bench_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    q_values = agent.policy_net(state_tensor).squeeze()
                    print(f"    - State '{name}':")
                    for i, q_val in enumerate(q_values):
                        action_name = " ".join(ACTIONS[i]) if isinstance(ACTIONS[i], list) else ACTIONS[i]
                        print(f"        - Q({action_name}): {q_val:.2f}")
            agent.policy_net.train()
        print("=" * 30 + "\n")

        writer.add_scalar('Performance/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Performance/Episode_Duration', episode_steps, episode)
        writer.add_scalar('Training/Average_Loss', avg_loss, episode)
        writer.add_scalar('Training/Epsilon', agent.epsilon, episode)

        if episode % TARGET_UPDATE_FREQUENCY == 0:
            print(">>> Updating target network...")
            agent.update_target_network()
            torch.save(agent.policy_net.state_dict(), f"dqn_mario_episode_{episode}.pth")
<<<<<<< HEAD


if __name__ == "__main__":
    run_dqn_training()
=======
>>>>>>> a54129cf58c1ab5fe6d5ef1f3c03ba1870c41dda
