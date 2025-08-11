# SNES AI

Este projeto utiliza Python para criar agentes de IA que podem jogar jogos de Super Nintendo (SNES), com foco principal em Super Mario World. Ele inclui um agente reativo simples e um agente mais complexo baseado em Deep Q-Learning (DQN).

## Funcionalidades

*   **Agente Reativo**: Um robô simples que usa reconhecimento de template para identificar e reagir a inimigos na tela.
*   **Agente DQN**: Um agente de aprendizado por reforço que aprende a jogar o jogo através de tentativa e erro, usando uma rede neural profunda para tomar decisões.
*   **Visualização do Jogo**: Scripts para visualizar o que o agente está "vendo" em tempo real.
*   **Interação com o Emulador**: Utiliza `pyautogui` para enviar comandos de teclado para o emulador de SNES (RetroArch) e `mss` para capturar a tela.

## Estrutura do Projeto

*   `dqn_executor.py`: O script principal para treinar o agente DQN.
*   `robo_reativo.py`: O script para executar o agente reativo baseado em template.
*   `agir_no_jogo.py`: Um script simples para testar a interação com o jogo.
*   `ver_jogo.py`: Um script para visualizar a tela do jogo que o agente vê.
*   `inimigo_template.png`: O template de imagem usado pelo agente reativo para detectar inimigos.
*   `runs/`: Diretório onde os logs do TensorBoard para o treinamento do DQN são salvos.

## Pré-requisitos

*   Python 3.x
*   Um emulador de SNES, como o [RetroArch](https://www.retroarch.com/)
*   O ROM do jogo (por exemplo, Super Mario World)

## Instalação

1.  Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/snes-ai.git
    cd snes-ai
    ```

2.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: Um arquivo `requirements.txt` precisa ser criado com as bibliotecas necessárias: `torch`, `numpy`, `opencv-python`, `mss`, `pyautogui`, `tensorboard`)*

## Como Usar

### 1. Configurar o Emulador

1.  Abra o RetroArch e carregue o ROM do jogo.
2.  Certifique-se de que a janela do jogo esteja visível e não minimizada.
3.  Ajuste as coordenadas da tela no script que você deseja executar (`dqn_executor.py`, `robo_reativo.py`, ou `ver_jogo.py`). As coordenadas estão na variável `bounding_box` ou `SCREEN_POS`.

### 2. Executar os Scripts

*   **Para testar a interação:**
    ```bash
    python agir_no_jogo.py
    ```

*   **Para visualizar a tela do jogo:**
    ```bash
    python ver_jogo.py
    ```

*   **Para executar o robô reativo:**
    ```bash
    python robo_reativo.py
    ```

*   **Para treinar o agente DQN:**
    ```bash
    python dqn_executor.py
    ```

## Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.