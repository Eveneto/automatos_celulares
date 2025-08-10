"""
Utilitários gerais para autômatos celulares elementares.

Este módulo contém funções auxiliares para geração de estados iniciais,
análise de padrões e outras operações úteis.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Callable
import json
import csv
from datetime import datetime


def gerar_estado_aleatorio(tamanho: int, densidade: float = 0.5, 
                          semente: Optional[int] = None) -> np.ndarray:
    """
    Gera um estado inicial aleatório.
    
    Args:
        tamanho: Número de células
        densidade: Proporção de células ativas (0.0 a 1.0)
        semente: Semente para reprodutibilidade
        
    Returns:
        Array numpy com estado inicial
    """
    if semente is not None:
        np.random.seed(semente)
    
    return np.random.choice([0, 1], size=tamanho, p=[1-densidade, densidade])


def gerar_estado_impulso(tamanho: int, posicao: Optional[int] = None) -> np.ndarray:
    """
    Gera estado com uma única célula ativa (impulso).
    
    Args:
        tamanho: Número de células
        posicao: Posição da célula ativa (se None, usa o centro)
        
    Returns:
        Array numpy com estado inicial
    """
    estado = np.zeros(tamanho, dtype=int)
    if posicao is None:
        posicao = tamanho // 2
    estado[posicao] = 1
    return estado


def gerar_estado_bloco(tamanho: int, tamanho_bloco: int, 
                      posicao: Optional[int] = None) -> np.ndarray:
    """
    Gera estado com um bloco de células ativas.
    
    Args:
        tamanho: Número total de células
        tamanho_bloco: Tamanho do bloco ativo
        posicao: Posição inicial do bloco (se None, centraliza)
        
    Returns:
        Array numpy com estado inicial
    """
    estado = np.zeros(tamanho, dtype=int)
    
    if posicao is None:
        posicao = (tamanho - tamanho_bloco) // 2
    
    fim = min(posicao + tamanho_bloco, tamanho)
    estado[posicao:fim] = 1
    
    return estado


def gerar_estado_periodico(tamanho: int, padrao: List[int]) -> np.ndarray:
    """
    Gera estado repetindo um padrão periodicamente.
    
    Args:
        tamanho: Número total de células
        padrao: Lista com o padrão a repetir
        
    Returns:
        Array numpy com estado inicial
    """
    estado = np.zeros(tamanho, dtype=int)
    
    for i in range(tamanho):
        estado[i] = padrao[i % len(padrao)]
    
    return estado


def detectar_simetria(estado: np.ndarray) -> Dict[str, bool]:
    """
    Detecta simetrias em um estado.
    
    Args:
        estado: Array com o estado a analisar
        
    Returns:
        Dicionário com tipos de simetria detectados
    """
    return {
        'reflexiva': np.array_equal(estado, estado[::-1]),
        'rotacional_180': np.array_equal(estado, 1 - estado[::-1]),
        'translacional': len(np.unique(estado)) == 1
    }


def calcular_distancia_hamming(estado1: np.ndarray, estado2: np.ndarray) -> int:
    """
    Calcula a distância de Hamming entre dois estados.
    
    Args:
        estado1: Primeiro estado
        estado2: Segundo estado
        
    Returns:
        Número de posições diferentes
    """
    if len(estado1) != len(estado2):
        raise ValueError("Estados devem ter o mesmo tamanho")
    
    return np.sum(estado1 != estado2)


def encontrar_padroes_locais(matriz: np.ndarray, tamanho_janela: int = 3) -> Dict:
    """
    Encontra padrões locais recorrentes na evolução.
    
    Args:
        matriz: Matriz de evolução (gerações x células)
        tamanho_janela: Tamanho da janela para análise
        
    Returns:
        Dicionário com padrões encontrados e suas frequências
    """
    padroes = {}
    
    for geracao in matriz:
        for i in range(len(geracao) - tamanho_janela + 1):
            padrao = tuple(geracao[i:i+tamanho_janela])
            padroes[padrao] = padroes.get(padrao, 0) + 1
    
    # Ordenar por frequência
    padroes_ordenados = dict(sorted(padroes.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'padroes': padroes_ordenados,
        'total_padroes': len(padroes_ordenados),
        'padrao_mais_comum': max(padroes_ordenados.items(), key=lambda x: x[1]) if padroes_ordenados else None
    }


def calcular_entropia(estado: np.ndarray) -> float:
    """
    Calcula a entropia de Shannon de um estado.
    
    Args:
        estado: Array com o estado
        
    Returns:
        Valor da entropia
    """
    # Contar frequências
    valores, counts = np.unique(estado, return_counts=True)
    probabilidades = counts / len(estado)
    
    # Calcular entropia
    entropia = -np.sum(probabilidades * np.log2(probabilidades + 1e-10))
    
    return entropia


def calcular_dimensao_fractal(matriz: np.ndarray, metodo: str = 'box_counting') -> float:
    """
    Estima a dimensão fractal de um padrão.
    
    Args:
        matriz: Matriz de evolução
        metodo: Método de cálculo ('box_counting')
        
    Returns:
        Estimativa da dimensão fractal
    """
    if metodo != 'box_counting':
        raise ValueError("Apenas 'box_counting' implementado")
    
    # Simplificação: usar apenas células ativas
    pontos_ativos = np.argwhere(matriz == 1)
    
    if len(pontos_ativos) == 0:
        return 0.0
    
    # Box counting simplificado
    tamanhos_caixa = [2, 4, 8, 16, 32]
    contagens = []
    
    altura, largura = matriz.shape
    
    for tamanho in tamanhos_caixa:
        if tamanho > min(altura, largura):
            break
        
        caixas_ocupadas = set()
        
        for ponto in pontos_ativos:
            y, x = ponto
            caixa_y = y // tamanho
            caixa_x = x // tamanho
            caixas_ocupadas.add((caixa_y, caixa_x))
        
        contagens.append(len(caixas_ocupadas))
    
    if len(contagens) < 2:
        return 0.0
    
    # Regressão linear em log-log
    log_tamanhos = np.log(tamanhos_caixa[:len(contagens)])
    log_contagens = np.log(contagens)
    
    # Dimensão fractal = -coeficiente angular
    coef = np.polyfit(log_tamanhos, log_contagens, 1)[0]
    
    return -coef


def exportar_dados(automato, formato: str = 'json', nome_arquivo: Optional[str] = None) -> str:
    """
    Exporta dados do autômato para arquivo.
    
    Args:
        automato: Instância de AutomatoElementar
        formato: Formato de exportação ('json', 'csv', 'numpy')
        nome_arquivo: Nome do arquivo (se None, gera automaticamente)
        
    Returns:
        Caminho do arquivo criado
    """
    if nome_arquivo is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"automato_regra_{automato.regra}_{timestamp}"
    
    if formato == 'json':
        dados = {
            'regra': automato.regra,
            'tamanho': automato.tamanho,
            'geracoes': len(automato.historico),
            'condicao_contorno': automato.condicao_contorno,
            'estadisticas': automato.obter_estatisticas(),
            'evolucao': [estado.tolist() for estado in automato.historico]
        }
        
        caminho = f"{nome_arquivo}.json"
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)
    
    elif formato == 'csv':
        matriz = automato.obter_matriz_evolucao()
        caminho = f"{nome_arquivo}.csv"
        
        with open(caminho, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Cabeçalho
            writer.writerow(['geracao'] + [f'celula_{i}' for i in range(automato.tamanho)])
            # Dados
            for i, estado in enumerate(matriz):
                writer.writerow([i] + estado.tolist())
    
    elif formato == 'numpy':
        matriz = automato.obter_matriz_evolucao()
        caminho = f"{nome_arquivo}.npy"
        np.save(caminho, matriz)
    
    else:
        raise ValueError("Formato deve ser 'json', 'csv' ou 'numpy'")
    
    return caminho


def carregar_dados(caminho: str):
    """
    Carrega dados de autômato de arquivo.
    
    Args:
        caminho: Caminho do arquivo
        
    Returns:
        Dados carregados
    """
    if caminho.endswith('.json'):
        with open(caminho, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif caminho.endswith('.csv'):
        dados = []
        with open(caminho, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Pular cabeçalho
            for row in reader:
                dados.append([int(x) for x in row[1:]])  # Pular coluna de geração
        return np.array(dados)
    
    elif caminho.endswith('.npy'):
        return np.load(caminho)
    
    else:
        raise ValueError("Formato de arquivo não suportado")


def analisar_convergencia(automato, tolerancia: float = 1e-6) -> Dict:
    """
    Analisa se o autômato convergiu para um estado estável.
    
    Args:
        automato: Instância de AutomatoElementar
        tolerancia: Tolerância para considerar convergência
        
    Returns:
        Informações sobre convergência
    """
    if len(automato.historico) < 2:
        return {'convergiu': False, 'geracao_convergencia': None}
    
    # Analisar últimas gerações para detectar estabilidade
    janela = min(10, len(automato.historico) // 2)
    
    if len(automato.historico) < janela:
        return {'convergiu': False, 'geracao_convergencia': None}
    
    ultimas_geracoes = automato.historico[-janela:]
    
    # Verificar se todas as gerações na janela são iguais
    primeira = ultimas_geracoes[0]
    for geracao in ultimas_geracoes[1:]:
        if not np.array_equal(primeira, geracao):
            return {'convergiu': False, 'geracao_convergencia': None}
    
    # Se chegou aqui, convergiu
    geracao_convergencia = len(automato.historico) - janela
    
    return {
        'convergiu': True,
        'geracao_convergencia': geracao_convergencia,
        'estado_final': primeira.copy(),
        'tipo_convergencia': 'estado_fixo'
    }


def benchmark_regras(regras: List[int], tamanho: int = 101, geracoes: int = 100) -> Dict:
    """
    Faz benchmark de múltiplas regras para comparação de performance.
    
    Args:
        regras: Lista de regras para testar
        tamanho: Tamanho da grade
        geracoes: Número de gerações
        
    Returns:
        Resultados do benchmark
    """
    import time
    try:
        from .automato_elementar import AutomatoElementar
    except ImportError:
        from automato_elementar import AutomatoElementar
    
    resultados = {}
    
    for regra in regras:
        inicio = time.time()
        
        automato = AutomatoElementar(regra, tamanho)
        automato.evoluir(geracoes)
        
        fim = time.time()
        tempo_execucao = fim - inicio
        
        stats = automato.obter_estatisticas()
        
        resultados[regra] = {
            'tempo_execucao': tempo_execucao,
            'densidade_final': stats['densidade_final'],
            'periodo_detectado': stats['periodo_detectado'],
            'geracoes_executadas': stats['geracoes']
        }
    
    return {
        'resultados_individuais': resultados,
        'tempo_total': sum(r['tempo_execucao'] for r in resultados.values()),
        'regra_mais_rapida': min(resultados.items(), key=lambda x: x[1]['tempo_execucao']),
        'regra_mais_lenta': max(resultados.items(), key=lambda x: x[1]['tempo_execucao'])
    }
