import retro
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

# O Stable-Retro requer um ficheiro de integração para cada jogo.
# O SuperMarioWorld-Snes já está incluído.
# O 'scenario.json' permite criar funções de recompensa baseadas na memória do jogo.
# Se não especificarmos um, a recompensa é baseada apenas no 'info' retornado (ex: pontuação).
GAME = "SuperMarioWorld-Snes"
STATE = "YoshiIsland1" # O nome do primeiro nível

def main():
    # Cria o ambiente do jogo. make_vec_env cria múltiplos ambientes
    # para treinar em paralelo, o que acelera muito o processo.
    # n_envs = 4 significa que 4 instâncias do jogo correrão em paralelo.
    vec_env = make_vec_env(
        lambda: retro.make(GAME, state=STATE), 
        n_envs=4, 
        seed=0
    )
    
    # Aplica wrappers para preparar as observações para a rede neural
    # Idêntico ao nosso process_frame e frame_stack, mas feito de forma otimizada.
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)
    
    # Inicializa o agente PPO (uma alternativa poderosa ao DQN)
    # CnnPolicy é uma política baseada em Redes Neurais Convolucionais, ideal para imagens.
    model = PPO(
        "CnnPolicy", 
        vec_env, 
        verbose=1,
        tensorboard_log="./ppo_mario_tensorboard/"
    )
    
    # Inicia o treino
    # O Stable-Baselines3 trata de todo o loop de treino, logging, etc.
    model.learn(total_timesteps=1_000_000)

    # Guarda o modelo treinado
    model.save("ppo_mario_model")

    # Para ver o seu agente a jogar
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        # vec_env.render() # Descomente para ver a janela do jogo

if __name__ == "__main__":
    main()