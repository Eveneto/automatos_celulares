"""
Exemplos de Regras Famosas de Autômatos Celulares Elementares

Este script demonstra o uso de algumas das regras mais conhecidas e
interessantes dos autômatos celulares elementares de Wolfram.
"""

import sys
import os

# Adicionar src ao path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from automato_elementar import AutomatoElementar
from visualizador import Visualizador
from classificador import ClassificadorWolfram
import matplotlib.pyplot as plt


def exemplo_regra_30():
    """
    Regra 30 - Comportamento Caótico
    
    A regra 30 é uma das mais famosas por gerar padrões aparentemente
    aleatórios. É usada como gerador de números pseudoaleatórios.
    """
    print("=== REGRA 30 - COMPORTAMENTO CAÓTICO ===")
    
    # Criar autômato
    automato = AutomatoElementar(regra=30, tamanho=101)
    automato.evoluir(geracoes=100)
    
    # Visualizar
    viz = Visualizador(automato)
    fig = viz.mostrar_evolucao()
    plt.title("Regra 30 - Padrão Caótico")
    plt.show()
    
    # Mostrar estatísticas
    stats = automato.obter_estatisticas()
    print(f"Densidade final: {stats['densidade_final']:.3f}")
    print(f"Período detectado: {stats['periodo_detectado']}")
    
    # Classificação
    classificador = ClassificadorWolfram()
    classe = classificador.classificar_regra(30)
    print(f"Classificação: {classe['nome_classe']}")
    print(f"Descrição: {classe['descricao']}")
    print()


def exemplo_regra_90():
    """
    Regra 90 - Triângulo de Sierpinski
    
    A regra 90 gera o famoso fractal conhecido como Triângulo de Sierpinski.
    """
    print("=== REGRA 90 - TRIÂNGULO DE SIERPINSKI ===")
    
    # Criar autômato
    automato = AutomatoElementar(regra=90, tamanho=101)
    automato.evoluir(geracoes=80)
    
    # Visualizar
    viz = Visualizador(automato)
    viz.definir_esquema_cor('azul')
    fig = viz.mostrar_evolucao()
    plt.title("Regra 90 - Triângulo de Sierpinski")
    plt.show()
    
    # Mostrar tabela da regra
    fig_regra = viz.mostrar_regra_binaria()
    plt.show()
    
    # Estatísticas
    stats = automato.obter_estatisticas()
    print(f"Densidade final: {stats['densidade_final']:.3f}")
    print(f"Período detectado: {stats['periodo_detectado']}")
    
    # Classificação
    classificador = ClassificadorWolfram()
    classe = classificador.classificar_regra(90)
    print(f"Classificação: {classe['nome_classe']}")
    print()


def exemplo_regra_110():
    """
    Regra 110 - Computação Universal
    
    A regra 110 é uma das poucas regras elementares que demonstram
    computação universal, sendo capaz de simular qualquer computação.
    """
    print("=== REGRA 110 - COMPUTAÇÃO UNIVERSAL ===")
    
    # Criar autômato com estado inicial mais interessante
    automato = AutomatoElementar(regra=110, tamanho=101)
    
    # Estado inicial: algumas células ativas espalhadas
    import numpy as np
    estado_inicial = np.zeros(101)
    estado_inicial[45:55] = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    automato.resetar(estado_inicial)
    
    automato.evoluir(geracoes=200)
    
    # Visualizar
    viz = Visualizador(automato)
    viz.definir_esquema_cor('verde')
    fig = viz.mostrar_evolucao()
    plt.title("Regra 110 - Estruturas Complexas")
    plt.show()
    
    # Análise de densidade
    fig_densidade = viz.plotar_densidade_temporal()
    plt.show()
    
    # Estatísticas
    stats = automato.obter_estatisticas()
    print(f"Densidade final: {stats['densidade_final']:.3f}")
    print(f"Período detectado: {stats['periodo_detectado']}")
    
    # Classificação
    classificador = ClassificadorWolfram()
    classe = classificador.classificar_regra(110)
    print(f"Classificação: {classe['nome_classe']}")
    print()


def exemplo_regra_150():
    """
    Regra 150 - Padrões Fractais
    
    A regra 150 produz padrões fractais interessantes e simétricos.
    """
    print("=== REGRA 150 - PADRÕES FRACTAIS ===")
    
    # Criar autômato
    automato = AutomatoElementar(regra=150, tamanho=101)
    automato.evoluir(geracoes=80)
    
    # Visualizar
    viz = Visualizador(automato)
    viz.definir_esquema_cor('roxo')
    fig = viz.mostrar_evolucao()
    plt.title("Regra 150 - Padrões Fractais")
    plt.show()
    
    # Estatísticas
    stats = automato.obter_estatisticas()
    print(f"Densidade final: {stats['densidade_final']:.3f}")
    print(f"Período detectado: {stats['periodo_detectado']}")
    print()


def exemplo_regra_184():
    """
    Regra 184 - Modelo de Tráfego
    
    A regra 184 pode ser interpretada como um modelo simples de tráfego,
    onde as células ativas representam carros.
    """
    print("=== REGRA 184 - MODELO DE TRÁFEGO ===")
    
    # Criar autômato com densidade média de "carros"
    automato = AutomatoElementar(regra=184, tamanho=101)
    
    # Estado inicial: distribuição aleatória de carros
    from src.utils import gerar_estado_aleatorio
    estado_inicial = gerar_estado_aleatorio(101, densidade=0.3, semente=42)
    automato.resetar(estado_inicial)
    
    automato.evoluir(geracoes=100)
    
    # Visualizar
    viz = Visualizador(automato)
    viz.definir_esquema_cor('vermelho')
    fig = viz.mostrar_evolucao()
    plt.title("Regra 184 - Fluxo de Tráfego")
    plt.show()
    
    # Criar animação
    print("Criando animação...")
    anim = viz.criar_animacao(intervalo=150)
    plt.show()
    
    # Estatísticas
    stats = automato.obter_estatisticas()
    print(f"Densidade final: {stats['densidade_final']:.3f}")
    print(f"Período detectado: {stats['periodo_detectado']}")
    print()


def comparacao_regras_por_classe():
    """
    Compara regras representativas de cada classe de Wolfram.
    """
    print("=== COMPARAÇÃO POR CLASSE DE WOLFRAM ===")
    
    # Regras representativas de cada classe
    regras_exemplo = {
        'Classe I (Homogêneo)': 8,
        'Classe II (Periódico)': 150,
        'Classe III (Caótico)': 30,
        'Classe IV (Complexo)': 110
    }
    
    # Criar autômato base para comparação
    automato_base = AutomatoElementar(regra=30, tamanho=101)  # Temporário
    viz = Visualizador(automato_base)
    
    # Comparar visualmente
    regras = list(regras_exemplo.values())
    fig = viz.comparar_regras(regras, geracoes=80)
    
    # Adicionar labels das classes
    axes = fig.get_axes()
    for i, (classe, regra) in enumerate(regras_exemplo.items()):
        axes[i].set_title(f'Regra {regra}\n{classe}', fontweight='bold')
    
    plt.suptitle('Comparação das Classes de Wolfram', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Análise estatística
    classificador = ClassificadorWolfram()
    
    print("Análise detalhada por classe:")
    for classe, regra in regras_exemplo.items():
        resultado = classificador.classificar_regra(regra)
        print(f"\n{classe}:")
        print(f"  Regra: {regra}")
        print(f"  Classificação: {resultado['nome_classe']}")
        print(f"  Confiança: {resultado.get('confianca', 'N/A')}")
        if 'metricas' in resultado:
            metricas = resultado['metricas']
            print(f"  Homogeneidade: {metricas['homogeneidade']:.3f}")
            print(f"  Complexidade: {metricas['complexidade']:.3f}")
            print(f"  Estabilidade: {metricas['estabilidade']:.3f}")


def main():
    """Executa todos os exemplos."""
    print("🔬 EXEMPLOS DE AUTÔMATOS CELULARES ELEMENTARES")
    print("=" * 50)
    
    try:
        # Executar exemplos individuais
        exemplo_regra_30()
        exemplo_regra_90()
        exemplo_regra_110()
        exemplo_regra_150()
        exemplo_regra_184()
        
        # Comparação final
        comparacao_regras_por_classe()
        
        print("✅ Todos os exemplos executados com sucesso!")
        print("\nPara explorar mais regras, experimente modificar os números")
        print("das regras nos exemplos acima (valores de 0 a 255).")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        print("Verifique se todas as dependências estão instaladas:")
        print("pip install numpy matplotlib pillow")


if __name__ == "__main__":
    main()
