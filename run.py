import sys
import os

# Add the 'src' directory to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """
    Main entry point to run the different agents and scripts.
    """
    print("Welcome to SNES AI!")
    print("Please choose which script to run:")
    print("1. Reactive Agent (Jumps from template-based enemies)")
    print("2. DQN Agent (Deep reinforcement learning training)")
    print("3. View Game Screen")
    print("4. Test Input Controls")

    choice = input("Enter the number of your choice: ")

    if choice == '1':
        print("\nRunning the Reactive Agent...")
        # We use the import here to avoid loading heavy modules unnecessarily
        from agents import reactive_agent
    elif choice == '2':
        print("\nRunning the DQN Agent...")
        from agents import dqn_agent
    elif choice == '3':
        print("\nRunning the Screen Visualization...")
        from scripts import view_game
    elif choice == '4':
        print("\nRunning the Input Test...")
        from scripts import test_input
    else:
        print("Invalid choice. Please run the script again and choose one of the options.")

if __name__ == "__main__":
    main()
