# -- SCREEN CAPTURE --
# The coordinates of the game screen to capture (top, left, width, height)
BOUNDING_BOX = {'top': 100, 'left': 100, 'width': 800, 'height': 600}

# -- KEY BINDINGS --
# The key to press to make the character jump
JUMP_KEY = 'z'

# -- ASSETS --
# The path to the enemy template image
ENEMY_TEMPLATE_PATH = 'assets/inimigo_template.png'

# -- AGENT CONFIG --
# The confidence threshold for template matching (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.5

# -- DQN AGENT --
# Actions the agent can take
ACTIONS = {
    0: 'right',             # Walk right
    1: ['right', 'x'],      # Run right
    2: 'z',                 # Jump (short)
    3: ['right', 'z'],      # Jump right (normal)
    4: ['right', 'x', 'z'], # Run and jump right (long)
    5: ['up', 'right'],     # Crucial for climbing ramps
}
ACTION_SPACE_SIZE = len(ACTIONS)

# Screen configuration
FRAME_STACK_SIZE = 4
INPUT_SHAPE = (FRAME_STACK_SIZE, 84, 84)

# DQN hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
LEARNING_RATE = 0.00025
MEMORY_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 10