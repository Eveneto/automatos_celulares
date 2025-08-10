"""
Script de demonstração interativa dos Autômatos Celulares Elementares

Execute este script para uma demonstração completa do projeto.
"""

import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from automato_elementar import AutomatoElementar
from visualizador import Visualizador
from classificador import ClassificadorWolfram
from utils import *
import matplotlib.pyplot as plt
import numpy as np


def demonstracao_basica():
    """Demonstração básica do funcionamento."""
    print("🎯 DEMONSTRAÇÃO BÁSICA")
    print("=" * 25)
    
    # Criar autômato com regra famosa
    print("Criando autômato com Regra 30 (comportamento caótico)...")
    automato = AutomatoElementar(regra=30, tamanho=101)
    
    print("Estado inicial:")
    print(automato)
    
    print("\nEvoluindo por 50 gerações...")
    automato.evoluir(50)
    
    print("Estado final:")
    print(automato)
    
    # Mostrar estatísticas
    stats = automato.obter_estatisticas()
    print(f"\nEstatísticas:")
    print(f"  Densidade inicial: {stats['densidade_inicial']:.3f}")
    print(f"  Densidade final: {stats['densidade_final']:.3f}")
    print(f"  Período detectado: {stats['periodo_detectado']}")
    
    # Visualizar
    print("\nVisualizando evolução...")
    viz = Visualizador(automato)
    fig = viz.mostrar_evolucao()
    plt.show()
    
    return automato


def demonstracao_classificacao():
    """Demonstração do sistema de classificação."""
    print("\n🏷️ SISTEMA DE CLASSIFICAÇÃO")
    print("=" * 30)
    
    classificador = ClassificadorWolfram()
    
    # Regras exemplo de cada classe
    regras_exemplo = [8, 150, 30, 110]
    nomes_exemplo = ["Homogêneo", "Periódico", "Caótico", "Complexo"]
    
    print("Classificando regras exemplo...")
    
    for regra, nome in zip(regras_exemplo, nomes_exemplo):
        resultado = classificador.classificar_regra(regra)
        print(f"\nRegra {regra} ({nome}):")
        print(f"  Classe: {resultado['nome_classe']}")
        print(f"  Descrição: {resultado['descricao']}")
        if 'confianca' in resultado:
            print(f"  Confiança: {resultado['confianca']:.2f}")


def demonstracao_comparativa():
    """Demonstração comparativa entre regras."""
    print("\n📊 COMPARAÇÃO ENTRE REGRAS")
    print("=" * 30)
    
    # Selecionar regras interessantes
    regras = [30, 90, 110, 150]
    
    # Criar autômato base para visualização
    automato_base = AutomatoElementar(30, 101)
    viz = Visualizador(automato_base)
    
    print("Comparando regras lado a lado...")
    fig = viz.comparar_regras(regras, geracoes=60)
    plt.suptitle('Comparação de Regras Famosas', fontsize=16)
    plt.show()
    
    return regras


def demonstracao_estados_iniciais():
    """Demonstração com diferentes estados iniciais."""
    print("\n🎲 DIFERENTES ESTADOS INICIAIS")
    print("=" * 33)
    
    regra = 110  # Regra complexa
    tamanho = 101
    
    estados = {
        'Impulso': gerar_estado_impulso(tamanho),
        'Aleatório': gerar_estado_aleatorio(tamanho, 0.2, 42),
        'Bloco': gerar_estado_bloco(tamanho, 20)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (nome, estado_inicial) in enumerate(estados.items()):
        print(f"Testando estado inicial: {nome}")
        
        automato = AutomatoElementar(regra, tamanho)
        automato.resetar(estado_inicial)
        automato.evoluir(80)
        
        matriz = automato.obter_matriz_evolucao()
        axes[i].imshow(matriz, cmap='viridis', aspect='auto')
        axes[i].set_title(f'{nome}\nDensidade: {automato.calcular_densidade():.3f}')
        axes[i].invert_yaxis()
    
    plt.suptitle(f'Regra {regra} com Diferentes Estados Iniciais', fontsize=14)
    plt.tight_layout()
    plt.show()


def demonstracao_interativa():
    """Demonstração interativa permitindo escolha do usuário."""
    print("\n🎮 DEMONSTRAÇÃO INTERATIVA")
    print("=" * 28)
    
    print("Escolha uma regra para explorar (0-255):")
    print("Sugestões: 30 (caótico), 90 (fractal), 110 (complexo), 150 (periódico)")
    
    try:
        regra = int(input("Digite o número da regra: "))
        if not 0 <= regra <= 255:
            print("Regra deve estar entre 0 e 255. Usando regra 30.")
            regra = 30
    except ValueError:
        print("Entrada inválida. Usando regra 30.")
        regra = 30
    
    print(f"\nExplorando Regra {regra}...")
    
    # Criar e evoluir autômato
    automato = AutomatoElementar(regra, 101)
    automato.evoluir(100)
    
    # Visualizar evolução
    viz = Visualizador(automato)
    fig = viz.mostrar_evolucao()
    plt.show()
    
    # Mostrar tabela da regra
    fig_regra = viz.mostrar_regra_binaria()
    plt.show()
    
    # Classificar
    classificador = ClassificadorWolfram()
    resultado = classificador.classificar_regra(regra)
    
    print(f"\nAnálise da Regra {regra}:")
    print(f"Classificação: {resultado['nome_classe']}")
    print(f"Descrição: {resultado['descricao']}")
    
    # Estatísticas
    stats = automato.obter_estatisticas()
    print(f"\nEstatísticas:")
    print(f"  Densidade inicial: {stats['densidade_inicial']:.3f}")
    print(f"  Densidade final: {stats['densidade_final']:.3f}")
    print(f"  Densidade média: {stats['densidade_media']:.3f}")
    print(f"  Período detectado: {stats['periodo_detectado']}")


def main():
    """Função principal da demonstração."""
    print("🤖 AUTÔMATOS CELULARES ELEMENTARES DE WOLFRAM")
    print("=" * 50)
    print("Demonstração interativa do projeto")
    print("=" * 50)
    
    try:
        # Sequência de demonstrações
        automato = demonstracao_basica()
        demonstracao_classificacao()
        demonstracao_comparativa()
        demonstracao_estados_iniciais()
        demonstracao_interativa()
        
        print("\n🎉 DEMONSTRAÇÃO CONCLUÍDA!")
        print("=" * 26)
        print("\nPara explorar mais:")
        print("- Execute 'python exemplos/regras_famosas.py' para ver exemplos específicos")
        print("- Execute 'python exemplos/experimentos.py' para análises avançadas")
        print("- Execute 'python testes/test_automato.py' para validar o código")
        print("\nDocumentação completa no README.md")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstração interrompida pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro durante a demonstração: {e}")
        print("Verifique se todas as dependências estão instaladas:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
