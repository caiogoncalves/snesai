"""
Enhanced SNES AI Player - Vision-Based Approach
Uses screen capture + reinforcement learning to play SNES games
No ROM integration needed - works with any emulator!
"""

import retro
import torch
import numpy as np
import cv2
from mss import mss
import pyautogui
import time
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

class ScreenCaptureEnv(gym.Env):
    """
    Custom Gym environment that captures the screen and simulates keyboard inputs.
    Works with any SNES emulator (RetroArch, SNES9x, etc.)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, screen_region={'top': 100, 'left': 100, 'width': 800, 'height': 600}):
        super().__init__()
        
        self.screen_region = screen_region
        self.sct = mss()
        
        # Action space: 9 possible actions
        # 0: No action, 1: Right, 2: Left, 3: Jump (Z), 4: Run (X)
        # 5: Right+Jump, 6: Right+Run, 7: Right+Run+Jump, 8: Up+Right (for ramps)
        self.action_space = spaces.Discrete(9)
        
        # Observation space: 84x84 grayscale image
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(84, 84, 1), 
            dtype=np.uint8
        )
        
        self.previous_frame = None
        self.step_count = 0
        self.max_steps = 1000
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.step_count = 0
        
        # Capture initial frame
        frame = self._capture_screen()
        self.previous_frame = frame
        
        print("\n🎮 Environment Reset! Make sure the game is at the starting position.")
        print("   You have 3 seconds to focus the emulator window...")
        time.sleep(3)
        
        return frame, {}
    
    def step(self, action):
        """Execute action and return observation"""
        # Execute the action
        self._execute_action(action)
        
        # Small delay to let the game respond
        time.sleep(0.05)
        
        # Capture new frame
        frame = self._capture_screen()
        
        # Calculate reward (simple pixel difference for now)
        reward = self._calculate_reward(frame)
        
        # Check if done
        self.step_count += 1
        done = self.step_count >= self.max_steps
        truncated = False
        
        self.previous_frame = frame
        
        return frame, reward, done, truncated, {}
    
    def _capture_screen(self):
        """Capture and process screen"""
        screenshot = np.array(self.sct.grab(self.screen_region))
        
        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Add channel dimension
        return resized.reshape(84, 84, 1)
    
    def _execute_action(self, action):
        """Execute keyboard action"""
        action_map = {
            0: [],                          # No action
            1: ['right'],                   # Move right
            2: ['left'],                    # Move left
            3: ['z'],                       # Jump
            4: ['x'],                       # Run
            5: ['right', 'z'],              # Jump right
            6: ['right', 'x'],              # Run right
            7: ['right', 'x', 'z'],         # Run + Jump right
            8: ['up', 'right'],             # Climb ramp
        }
        
        keys = action_map.get(action, [])
        
        if keys:
            # Press all keys
            for key in keys:
                pyautogui.keyDown(key)
            
            time.sleep(0.05)
            
            # Release all keys
            for key in keys:
                pyautogui.keyUp(key)
    
    def _calculate_reward(self, current_frame):
        """
        Calculate reward based on frame difference.
        More movement = higher reward (encourages exploration)
        """
        if self.previous_frame is None:
            return 0.0
        
        # Calculate pixel difference
        diff = np.abs(current_frame.astype(float) - self.previous_frame.astype(float))
        movement = np.mean(diff)
        
        # Reward movement (exploration)
        reward = movement / 10.0
        
        # Small time penalty to encourage speed
        reward -= 0.01
        
        return reward
    
    def render(self):
        """Render is handled by the emulator"""
        pass
    
    def close(self):
        """Cleanup"""
        pass


class TrainingCallback(BaseCallback):
    """Custom callback for logging training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        # Log episode statistics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    
                    if len(self.episode_rewards) % 10 == 0:
                        avg_reward = np.mean(self.episode_rewards[-10:])
                        avg_length = np.mean(self.episode_lengths[-10:])
                        print(f"\n📊 Episodes: {len(self.episode_rewards)} | "
                              f"Avg Reward: {avg_reward:.2f} | "
                              f"Avg Length: {avg_length:.0f}")
        
        return True


def train_vision_based_agent(total_timesteps=100000):
    """
    Train an agent using screen capture (no ROM needed!)
    """
    print("="*60)
    print("🎮 Vision-Based SNES AI Training")
    print("="*60)
    print("\nThis will train an AI to play SNES games by watching your screen!")
    print("\nSetup Instructions:")
    print("1. Open your SNES emulator (RetroArch, SNES9x, etc.)")
    print("2. Load Super Mario World")
    print("3. Position the game window at the configured location")
    print("4. Make sure the game is paused at the start of a level")
    print("\nPress ENTER when ready...")
    input()
    
    # Create environment
    print("\n🔧 Creating environment...")
    env = ScreenCaptureEnv(
        screen_region={'top': 100, 'left': 100, 'width': 800, 'height': 600}
    )
    
    # Wrap environment
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    
    # Create PPO agent
    print("🤖 Initializing PPO agent...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./ppo_vision_tensorboard/"
    )
    
    # Create callback
    callback = TrainingCallback()
    
    # Train
    print(f"\n🚀 Starting training for {total_timesteps} timesteps...")
    print("   (Press Ctrl+C to stop training and save the model)\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Save model
    model_path = "ppo_snes_vision_model"
    model.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    return model


def play_with_trained_model(model_path="ppo_snes_vision_model"):
    """
    Watch the trained agent play!
    """
    print("="*60)
    print("🎮 Playing with Trained Model")
    print("="*60)
    print("\nSetup Instructions:")
    print("1. Open your SNES emulator")
    print("2. Load Super Mario World")
    print("3. Position the game window")
    print("4. Start at the beginning of a level")
    print("\nPress ENTER when ready...")
    input()
    
    # Load environment and model
    env = ScreenCaptureEnv()
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    
    model = PPO.load(model_path)
    
    # Play
    obs = env.reset()
    print("\n🎮 AI is now playing! Press Ctrl+C to stop.\n")
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            
            if dones[0]:
                print("Episode finished! Resetting...")
                obs = env.reset()
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    
    env.close()


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("🎮 SNES AI - Vision-Based Training")
    print("="*60)
    print("\nChoose an option:")
    print("1. Train a new model")
    print("2. Watch a trained model play")
    print("3. Quick test (1000 steps)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        train_vision_based_agent(total_timesteps=100000)
    elif choice == "2":
        play_with_trained_model()
    elif choice == "3":
        train_vision_based_agent(total_timesteps=1000)
    else:
        print("Invalid choice!")
