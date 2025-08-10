"""
Experimentos e Análises Avançadas com Autômatos Celulares Elementares

Este script demonstra experimentos mais avançados, incluindo análise
estatística, detecção de padrões e estudos comparativos.
"""

import sys
import os

# Adicionar src ao path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from automato_elementar import AutomatoElementar
from visualizador import Visualizador
from classificador import ClassificadorWolfram
from utils import *
import matplotlib.pyplot as plt
import numpy as np


def experimento_estados_iniciais():
    """
    Analisa como diferentes estados iniciais afetam a evolução da mesma regra.
    """
    print("🧪 EXPERIMENTO: Efeito de Estados Iniciais Diferentes")
    print("=" * 50)
    
    regra = 30  # Regra caótica
    tamanho = 101
    geracoes = 80
    
    # Diferentes estados iniciais
    estados = {
        'Impulso Central': gerar_estado_impulso(tamanho),
        'Impulso Lateral': gerar_estado_impulso(tamanho, 20),
        'Bloco Central': gerar_estado_bloco(tamanho, 10),
        'Aleatório (30%)': gerar_estado_aleatorio(tamanho, 0.3, 42),
        'Padrão Periódico': gerar_estado_periodico(tamanho, [1, 0, 1, 0])
    }
    
    fig, axes = plt.subplots(1, len(estados), figsize=(20, 6))
    
    for i, (nome, estado_inicial) in enumerate(estados.items()):
        # Criar e evoluir autômato
        automato = AutomatoElementar(regra, tamanho)
        automato.resetar(estado_inicial)
        automato.evoluir(geracoes)
        
        # Plotar
        matriz = automato.obter_matriz_evolucao()
        axes[i].imshow(matriz, cmap='Blues', interpolation='nearest', aspect='auto')
        axes[i].set_title(f'{nome}\nDensidade final: {automato.calcular_densidade():.3f}')
        axes[i].set_xlabel('Posição')
        if i == 0:
            axes[i].set_ylabel('Geração')
        axes[i].invert_yaxis()
    
    plt.suptitle(f'Regra {regra} com Diferentes Estados Iniciais', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("Observação: Mesmo regras caóticas podem mostrar sensibilidade")
    print("ao estado inicial, mas tendem a convergir para comportamentos similares.\n")


def experimento_convergencia():
    """
    Analisa a convergência de diferentes regras ao longo do tempo.
    """
    print("📈 EXPERIMENTO: Análise de Convergência")
    print("=" * 40)
    
    regras_teste = [8, 32, 150, 184]  # Regras com diferentes comportamentos
    tamanho = 101
    geracoes = 150
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, regra in enumerate(regras_teste):
        automato = AutomatoElementar(regra, tamanho)
        automato.evoluir(geracoes)
        
        # Analisar convergência
        convergencia = analisar_convergencia(automato)
        
        # Plotar densidade ao longo do tempo
        densidades = [np.mean(estado) for estado in automato.historico]
        axes[i].plot(densidades, linewidth=2)
        axes[i].set_title(f'Regra {regra}\nConvergiu: {convergencia["convergiu"]}')
        axes[i].set_xlabel('Geração')
        axes[i].set_ylabel('Densidade')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1)
        
        if convergencia['convergiu']:
            conv_gen = convergencia['geracao_convergencia']
            axes[i].axvline(x=conv_gen, color='red', linestyle='--', 
                          label=f'Convergência (gen {conv_gen})')
            axes[i].legend()
        
        print(f"Regra {regra}: {'Convergiu' if convergencia['convergiu'] else 'Não convergiu'}")
        if convergencia['convergiu']:
            print(f"  - Geração de convergência: {convergencia['geracao_convergencia']}")
    
    plt.suptitle('Análise de Convergência por Regra', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print()


def experimento_deteccao_padroes():
    """
    Detecta e analisa padrões locais em diferentes regras.
    """
    print("🔍 EXPERIMENTO: Detecção de Padrões Locais")
    print("=" * 42)
    
    regras_teste = [30, 90, 110, 150]
    
    for regra in regras_teste:
        print(f"\nRegra {regra}:")
        
        # Criar e evoluir autômato
        automato = AutomatoElementar(regra, 101)
        automato.evoluir(100)
        
        # Detectar padrões
        matriz = automato.obter_matriz_evolucao()
        padroes = encontrar_padroes_locais(matriz, tamanho_janela=3)
        
        print(f"  Total de padrões únicos: {padroes['total_padroes']}")
        
        if padroes['padrao_mais_comum']:
            padrao, freq = padroes['padrao_mais_comum']
            padrao_str = ''.join(map(str, padrao))
            print(f"  Padrão mais comum: '{padrao_str}' (freq: {freq})")
        
        # Mostrar top 5 padrões
        print("  Top 5 padrões:")
        for j, (padrao, freq) in enumerate(list(padroes['padroes'].items())[:5]):
            padrao_str = ''.join(map(str, padrao))
            print(f"    {j+1}. '{padrao_str}': {freq} ocorrências")


def experimento_dimensao_fractal():
    """
    Calcula a dimensão fractal de padrões gerados.
    """
    print("📐 EXPERIMENTO: Dimensão Fractal")
    print("=" * 32)
    
    regras_fractais = [30, 90, 150, 110]
    
    print("Estimativas de dimensão fractal:")
    for regra in regras_fractais:
        automato = AutomatoElementar(regra, 101)
        automato.evoluir(100)
        
        matriz = automato.obter_matriz_evolucao()
        dim_fractal = calcular_dimensao_fractal(matriz)
        
        print(f"  Regra {regra}: {dim_fractal:.3f}")
    
    print("\nNota: Valores próximos a 2.0 indicam preenchimento")
    print("bidimensional, enquanto valores menores sugerem estruturas fractais.\n")


def experimento_benchmark_performance():
    """
    Testa performance de diferentes regras.
    """
    print("⚡ EXPERIMENTO: Benchmark de Performance")
    print("=" * 38)
    
    # Regras representativas de cada classe
    regras_teste = [8, 30, 90, 110, 150, 184]
    
    print("Executando benchmark...")
    resultados = benchmark_regras(regras_teste, tamanho=201, geracoes=200)
    
    print("\nResultados:")
    print(f"Tempo total: {resultados['tempo_total']:.3f} segundos")
    
    regra_rapida, dados_rapida = resultados['regra_mais_rapida']
    regra_lenta, dados_lenta = resultados['regra_mais_lenta']
    
    print(f"Regra mais rápida: {regra_rapida} ({dados_rapida['tempo_execucao']:.4f}s)")
    print(f"Regra mais lenta: {regra_lenta} ({dados_lenta['tempo_execucao']:.4f}s)")
    
    print("\nDetalhes por regra:")
    for regra, dados in resultados['resultados_individuais'].items():
        print(f"  Regra {regra}: {dados['tempo_execucao']:.4f}s, "
              f"densidade final: {dados['densidade_final']:.3f}")


def experimento_classificacao_massiva():
    """
    Classifica todas as 256 regras e analisa estatísticas.
    """
    print("📊 EXPERIMENTO: Classificação Massiva")
    print("=" * 35)
    
    print("Classificando todas as 256 regras...")
    
    classificador = ClassificadorWolfram()
    stats = classificador.obter_estatisticas_classificacao(list(range(0, 256, 10)))  # Sample
    
    print(f"\nEstatísticas (amostra de {stats['total_regras']} regras):")
    print(f"Total analisado: {stats['total_regras']}")
    
    for classe in [1, 2, 3, 4]:
        count = stats['por_classe'][classe]
        percent = stats['percentuais'][classe]
        nome = classificador._nome_classe(classe)
        print(f"{nome}: {count} regras ({percent:.1f}%)")
    
    # Plotar distribuição
    classes = ['Classe I', 'Classe II', 'Classe III', 'Classe IV']
    counts = [stats['por_classe'][i+1] for i in range(4)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    plt.title('Distribuição das Classes de Wolfram (Amostra)', fontweight='bold')
    plt.ylabel('Número de Regras')
    plt.xlabel('Classe')
    
    # Adicionar valores nas barras
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\nExemplos de regras por classe:")
    for classe in [1, 2, 3, 4]:
        regras = stats['regras_por_classe'][classe][:5]  # Primeiras 5
        nome = classificador._nome_classe(classe)
        print(f"{nome}: {regras}")


def experimento_simetrias():
    """
    Analisa simetrias em padrões gerados.
    """
    print("🪞 EXPERIMENTO: Análise de Simetrias")
    print("=" * 33)
    
    regras_teste = [90, 150, 102, 170]
    
    for regra in regras_teste:
        print(f"\nRegra {regra}:")
        
        automato = AutomatoElementar(regra, 101)
        automato.evoluir(50)
        
        # Analisar simetrias no estado final
        simetrias = detectar_simetria(automato.estado_atual)
        
        print(f"  Simetria reflexiva: {'Sim' if simetrias['reflexiva'] else 'Não'}")
        print(f"  Simetria rotacional 180°: {'Sim' if simetrias['rotacional_180'] else 'Não'}")
        print(f"  Simetria translacional: {'Sim' if simetrias['translacional'] else 'Não'}")


def main():
    """Executa todos os experimentos."""
    print("🔬 EXPERIMENTOS AVANÇADOS COM AUTÔMATOS CELULARES")
    print("=" * 55)
    
    try:
        experimento_estados_iniciais()
        experimento_convergencia()
        experimento_deteccao_padroes()
        experimento_dimensao_fractal()
        experimento_benchmark_performance()
        experimento_classificacao_massiva()
        experimento_simetrias()
        
        print("\n✅ Todos os experimentos executados com sucesso!")
        print("\nEstes experimentos demonstram a riqueza e complexidade")
        print("dos autômatos celulares elementares, mostrando como regras")
        print("simples podem gerar comportamentos surpreendentemente diversos.")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
