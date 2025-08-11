import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from mss import mss
import pyautogui
import time
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# --- CONFIGURAÇÕES E HIPERPARÂMETROS ---

# AÇÕES ATUALIZADAS PARA LIDAR COM RAMPAS
ACTIONS = {
    0: 'right',             # Andar para a direita
    1: ['right', 'x'],      # Correr para a direita (afeta pulos)
    2: 'z',                 # Pular parado (pulo curto)
    3: ['right', 'z'],      # Pular para a direita (pulo normal)
    4: ['right', 'x', 'z'], # Correr e pular para a direita (pulo longo)
    5: ['up', 'right'],     # Ação crucial para subir rampas!
}
ACTION_SPACE_SIZE = len(ACTIONS)

# Configurações da tela
SCREEN_POS = {'top': 100, 'left': 100, 'width': 800, 'height': 600} # AJUSTE PARA A SUA TELA
FRAME_STACK_SIZE = 4
INPUT_SHAPE = (FRAME_STACK_SIZE, 84, 84)

# Hiperparâmetros do DQN
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
LEARNING_RATE = 0.00025
MEMORY_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 10

# --- FUNÇÃO DE PROCESSAMENTO DE IMAGEM ---

def process_frame(frame):
    """Converte um frame para escala de cinza e redimensiona."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)

# --- ARQUITETURA DA REDE NEURAL (CNN) ---

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255.0
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# --- CLASSE DO AGENTE DQN ---

class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"A usar o dispositivo: {self.device}")
        
        self.policy_net = DQN(input_shape, num_actions).to(self.device)
        self.target_net = DQN(input_shape, num_actions).to(self.device)
        self.update_target_network()
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.num_actions = num_actions

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.num_actions)
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_net(state_tensor)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()

# --- LOOP PRINCIPAL ---

if __name__ == "__main__":
    agent = DQNAgent(INPUT_SHAPE, ACTION_SPACE_SIZE)
    sct = mss()
    writer = SummaryWriter(f"runs/mario_dqn_{int(time.time())}")

    try:
        benchmark_images = {
            "Inicio": process_frame(cv2.imread("benchmark_inicio.png")),
            "Inimigo": process_frame(cv2.imread("benchmark_inimigo.png"))
        }
        print("Imagens de benchmark carregadas com sucesso.")
    except Exception as e:
        print(f"AVISO: Não foi possível carregar as imagens de benchmark. Erro: {e}")
        benchmark_images = None

    frame = sct.grab(SCREEN_POS)
    processed_frame = process_frame(np.array(frame))
    state_stack = deque([processed_frame] * FRAME_STACK_SIZE, maxlen=FRAME_STACK_SIZE)
    current_state = np.array(state_stack)

    episode = 0
    total_steps = 0

    print("A iniciar o treino em 5 segundos. Clique na janela do jogo!")
    time.sleep(5)

    while True:
        episode += 1
        episode_reward = 0
        episode_steps = 0
        
        episode_losses = []
        action_counts = {i: 0 for i in range(ACTION_SPACE_SIZE)}

        print(f"\n--- Episódio {episode} | Epsilon: {agent.epsilon:.4f} ---")
        # AQUI VOCÊ DEVE RESETAR O JOGO MANUALMENTE OU AUTOMATIZAR
        time.sleep(2)

        done = False
        while not done:
            action_idx = agent.act(current_state)
            action_keys = ACTIONS[action_idx]
            
            action_counts[action_idx] += 1

            if isinstance(action_keys, list):
                for key in action_keys: pyautogui.keyDown(key)
                time.sleep(0.05)
                for key in action_keys: pyautogui.keyUp(key)
            else:
                pyautogui.press(action_keys)

            # Lógica de recompensa (AINDA PRECISA SER MELHORADA)
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

        # Imprimir o relatório de final de episódio
        # ... (código do relatório exatamente como na resposta anterior)
        print("\n" + "="*30)
        print(f"RELATÓRIO DO EPISÓDIO {episode}")
        print("="*30)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        print(f"  Performance:\n    - Recompensa Total: {episode_reward:.2f}\n    - Duração: {episode_steps} passos")
        print(f"  Treino:\n    - Loss Média: {avg_loss:.5f}")
        total_actions = sum(action_counts.values())
        print(f"  Comportamento (Distribuição de Ações):")
        for idx, count in action_counts.items():
            action_name = " ".join(ACTIONS[idx]) if isinstance(ACTIONS[idx], list) else ACTIONS[idx]
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            print(f"    - Ação '{action_name}': {count} vezes ({percentage:.1f}%)")
        if benchmark_images:
            print(f"  Análise de Q-Values ('Confiança' da Rede):")
            agent.policy_net.eval()
            with torch.no_grad():
                for name, img in benchmark_images.items():
                    bench_state = np.array([img] * FRAME_STACK_SIZE)
                    state_tensor = torch.tensor(bench_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    q_values = agent.policy_net(state_tensor).squeeze()
                    print(f"    - Estado '{name}':")
                    for i, q_val in enumerate(q_values):
                        action_name = " ".join(ACTIONS[i]) if isinstance(ACTIONS[i], list) else ACTIONS[i]
                        print(f"        - Q({action_name}): {q_val:.2f}")
            agent.policy_net.train()
        print("="*30 + "\n")

        writer.add_scalar('Performance/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Performance/Episode_Duration', episode_steps, episode)
        writer.add_scalar('Training/Average_Loss', avg_loss, episode)
        writer.add_scalar('Training/Epsilon', agent.epsilon, episode)
        
        if agent.epsilon > EPSILON_END:
            agent.epsilon *= EPSILON_DECAY

        if episode % TARGET_UPDATE_FREQUENCY == 0:
            print(">>> A atualizar a rede alvo...")
            agent.update_target_network()
            torch.save(agent.policy_net.state_dict(), f"dqn_mario_episode_{episode}.pth")