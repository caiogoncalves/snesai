import sys
import os

# Adiciona o diretório 'src' ao path para que possamos importar os módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """
    Ponto de entrada principal para executar os diferentes agentes e scripts.
    """
    print("Bem-vindo ao SNES AI!")
    print("Por favor, escolha qual script executar:")
    print("1. Agente Reativo (Pula de inimigos baseados em template)")
    print("2. Agente DQN (Treino de aprendizado por reforço profundo)")
    print("3. Visualizar Tela do Jogo")
    print("4. Testar Controles de Input")

    choice = input("Digite o número da sua escolha: ")

    if choice == '1':
        print("\nA executar o Agente Reativo...")
        # Usamos o import aqui para evitar carregar módulos pesados desnecessariamente
        from agents import reactive_agent
    elif choice == '2':
        print("\nA executar o Agente DQN...")
        from agents import dqn_agent
    elif choice == '3':
        print("\nA executar a Visualização da Tela...")
        from scripts import view_game
    elif choice == '4':
        print("\nA executar o Teste de Input...")
        from scripts import test_input
    else:
        print("Escolha inválida. Por favor, execute o script novamente e escolha uma das opções.")

if __name__ == "__main__":
    main()
