"""
Classificador de Autômatos Celulares segundo Wolfram

Este módulo implementa a classificação de Wolfram para autômatos celulares
elementares em quatro classes baseadas em seu comportamento emergente.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from .automato_elementar import AutomatoElementar
except ImportError:
    from automato_elementar import AutomatoElementar


class ClassificadorWolfram:
    """
    Classifica autômatos celulares elementares segundo a taxonomia de Wolfram.
    
    Classes de Wolfram:
    - Classe I: Evolui para estado homogêneo
    - Classe II: Evolui para estruturas simples e periódicas
    - Classe III: Comportamento caótico
    - Classe IV: Estruturas complexas localizadas
    """
    
    # Classificação conhecida de algumas regras famosas
    CLASSIFICACAO_CONHECIDA = {
        # Classe I - Homogêneo
        0: 1, 8: 1, 32: 1, 40: 1, 128: 1, 136: 1, 160: 1, 168: 1,
        
        # Classe II - Periódico
        1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 9: 2, 10: 2, 11: 2,
        12: 2, 13: 2, 14: 2, 15: 2, 19: 2, 23: 2, 24: 2, 25: 2, 26: 2,
        27: 2, 28: 2, 29: 2, 31: 2, 33: 2, 34: 2, 35: 2, 36: 2, 37: 2,
        38: 2, 39: 2, 50: 2, 51: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2,
        62: 2, 90: 2, 94: 2, 102: 2, 150: 2, 154: 2, 158: 2, 178: 2,
        184: 2, 188: 2, 190: 2, 194: 2, 198: 2, 206: 2, 218: 2, 220: 2,
        222: 2, 250: 2,
        
        # Classe III - Caótico
        18: 3, 22: 3, 30: 3, 45: 3, 60: 3, 73: 3, 75: 3, 86: 3, 89: 3,
        101: 3, 105: 3, 106: 3, 109: 3, 120: 3, 122: 3, 124: 3, 129: 3,
        131: 3, 133: 3, 135: 3, 137: 3, 139: 3, 141: 3, 149: 3, 151: 3,
        161: 3, 163: 3, 165: 3, 167: 3, 169: 3, 171: 3, 182: 3, 183: 3,
        195: 3, 225: 3,
        
        # Classe IV - Complexo
        41: 4, 54: 4, 110: 4, 124: 4, 137: 4, 193: 4
    }
    
    def __init__(self):
        """Inicializa o classificador."""
        self.cache_classificacao = {}
    
    def classificar_regra(self, regra: int, tamanho: int = 101, geracoes: int = 200,
                         usar_cache: bool = True) -> Dict:
        """
        Classifica uma regra específica.
        
        Args:
            regra: Número da regra (0-255)
            tamanho: Tamanho da grade para análise
            geracoes: Número de gerações para análise
            usar_cache: Se deve usar classificação conhecida
            
        Returns:
            Dicionário com informações da classificação
        """
        if usar_cache and regra in self.CLASSIFICACAO_CONHECIDA:
            classe = self.CLASSIFICACAO_CONHECIDA[regra]
            return {
                'regra': regra,
                'classe': classe,
                'nome_classe': self._nome_classe(classe),
                'descricao': self._descricao_classe(classe),
                'fonte': 'literatura',
                'confianca': 1.0
            }
        
        # Análise computacional
        automato = AutomatoElementar(regra, tamanho)
        automato.evoluir(geracoes)
        
        resultado = self._analisar_comportamento(automato)
        resultado['regra'] = regra
        resultado['fonte'] = 'analise'
        
        return resultado
    
    def _analisar_comportamento(self, automato: AutomatoElementar) -> Dict:
        """
        Analisa o comportamento de um autômato para classificação.
        
        Args:
            automato: Instância do autômato evoluído
            
        Returns:
            Dicionário com análise comportamental
        """
        matriz = automato.obter_matriz_evolucao()
        
        # Métricas para classificação
        homogeneidade = self._calcular_homogeneidade(matriz)
        periodicidade = self._detectar_periodicidade(automato)
        complexidade = self._calcular_complexidade(matriz)
        estabilidade = self._calcular_estabilidade(matriz)
        
        # Lógica de classificação
        classe, confianca = self._determinar_classe(
            homogeneidade, periodicidade, complexidade, estabilidade
        )
        
        return {
            'classe': classe,
            'nome_classe': self._nome_classe(classe),
            'descricao': self._descricao_classe(classe),
            'confianca': confianca,
            'metricas': {
                'homogeneidade': homogeneidade,
                'periodicidade': periodicidade,
                'complexidade': complexidade,
                'estabilidade': estabilidade
            }
        }
    
    def _calcular_homogeneidade(self, matriz: np.ndarray) -> float:
        """
        Calcula o grau de homogeneidade do padrão final.
        
        Args:
            matriz: Matriz de evolução
            
        Returns:
            Valor entre 0 (heterogêneo) e 1 (homogêneo)
        """
        if len(matriz) < 10:
            return 0.0
        
        # Analisar últimas 10 gerações
        ultimas_geracoes = matriz[-10:]
        
        # Calcular variância média das últimas gerações
        variancias = [np.var(geracao) for geracao in ultimas_geracoes]
        variancia_media = np.mean(variancias)
        
        # Normalizar (0 = homogêneo, 0.25 = máximo teórico para variância binária)
        homogeneidade = 1.0 - min(variancia_media / 0.25, 1.0)
        
        return homogeneidade
    
    def _detectar_periodicidade(self, automato: AutomatoElementar) -> Dict:
        """
        Detecta padrões periódicos na evolução.
        
        Args:
            automato: Instância do autômato
            
        Returns:
            Dicionário com informações de periodicidade
        """
        periodo = automato.detectar_periodo(janela_busca=50)
        
        if periodo is not None:
            return {
                'periodo': periodo,
                'eh_periodico': True,
                'tipo': 'estrito' if periodo <= 5 else 'longo'
            }
        
        # Verificar quasi-periodicidade
        matriz = automato.obter_matriz_evolucao()
        if len(matriz) > 20:
            quasi_periodo = self._detectar_quasi_periodicidade(matriz)
            if quasi_periodo:
                return {
                    'periodo': quasi_periodo,
                    'eh_periodico': True,
                    'tipo': 'quasi'
                }
        
        return {
            'periodo': None,
            'eh_periodico': False,
            'tipo': 'aperiodico'
        }
    
    def _detectar_quasi_periodicidade(self, matriz: np.ndarray) -> Optional[int]:
        """
        Detecta quasi-periodicidade analisando correlações.
        
        Args:
            matriz: Matriz de evolução
            
        Returns:
            Período detectado ou None
        """
        if len(matriz) < 40:
            return None
        
        # Analisar últimas 30 gerações
        ultimas = matriz[-30:]
        
        # Procurar padrões similares
        for periodo in range(2, 15):
            correlacoes = []
            for i in range(len(ultimas) - periodo):
                corr = np.corrcoef(ultimas[i], ultimas[i + periodo])[0, 1]
                if not np.isnan(corr):
                    correlacoes.append(corr)
            
            if correlacoes and np.mean(correlacoes) > 0.8:
                return periodo
        
        return None
    
    def _calcular_complexidade(self, matriz: np.ndarray) -> float:
        """
        Calcula uma medida de complexidade baseada na entropia.
        
        Args:
            matriz: Matriz de evolução
            
        Returns:
            Valor de complexidade normalizado
        """
        if len(matriz) < 2:
            return 0.0
        
        # Calcular entropia das transições
        entropias = []
        for i in range(1, len(matriz)):
            diferenca = np.abs(matriz[i] - matriz[i-1])
            transicoes = np.sum(diferenca)
            total_celulas = len(matriz[i])
            
            if total_celulas > 0:
                entropia = transicoes / total_celulas
                entropias.append(entropia)
        
        if not entropias:
            return 0.0
        
        # Complexidade como variabilidade da entropia
        complexidade = np.std(entropias) if len(entropias) > 1 else np.mean(entropias)
        
        return min(complexidade, 1.0)
    
    def _calcular_estabilidade(self, matriz: np.ndarray) -> float:
        """
        Calcula a estabilidade temporal do padrão.
        
        Args:
            matriz: Matriz de evolução
            
        Returns:
            Valor de estabilidade (0 = instável, 1 = estável)
        """
        if len(matriz) < 10:
            return 0.0
        
        # Dividir em duas metades
        meio = len(matriz) // 2
        primeira_metade = matriz[:meio]
        segunda_metade = matriz[meio:]
        
        # Calcular densidade média de cada metade
        densidade1 = np.mean([np.mean(geracao) for geracao in primeira_metade])
        densidade2 = np.mean([np.mean(geracao) for geracao in segunda_metade])
        
        # Estabilidade baseada na diferença de densidades
        diferenca = abs(densidade1 - densidade2)
        estabilidade = 1.0 - min(diferenca, 1.0)
        
        return estabilidade
    
    def _determinar_classe(self, homogeneidade: float, periodicidade: Dict,
                          complexidade: float, estabilidade: float) -> Tuple[int, float]:
        """
        Determina a classe de Wolfram baseada nas métricas.
        
        Args:
            homogeneidade: Grau de homogeneidade
            periodicidade: Informações de periodicidade
            complexidade: Medida de complexidade
            estabilidade: Medida de estabilidade
            
        Returns:
            Tupla (classe, confiança)
        """
        # Classe I: Homogêneo
        if homogeneidade > 0.9 and estabilidade > 0.8:
            return 1, 0.9
        
        # Classe II: Periódico
        if periodicidade['eh_periodico']:
            if periodicidade['tipo'] in ['estrito', 'quasi']:
                return 2, 0.85
        
        # Classe III vs IV: baseado em complexidade e estabilidade
        if complexidade > 0.3:
            if estabilidade < 0.4:  # Alta complexidade, baixa estabilidade = caótico
                return 3, 0.7
            else:  # Alta complexidade, média/alta estabilidade = complexo
                return 4, 0.6
        
        # Classificação conservadora para casos ambíguos
        if estabilidade > 0.6:
            return 2, 0.5  # Provavelmente periódico
        else:
            return 3, 0.5  # Provavelmente caótico
    
    def _nome_classe(self, classe: int) -> str:
        """Retorna o nome da classe."""
        nomes = {
            1: "Classe I - Homogêneo",
            2: "Classe II - Periódico", 
            3: "Classe III - Caótico",
            4: "Classe IV - Complexo"
        }
        return nomes.get(classe, "Desconhecido")
    
    def _descricao_classe(self, classe: int) -> str:
        """Retorna a descrição da classe."""
        descricoes = {
            1: "Evolui rapidamente para um estado homogêneo (todas as células no mesmo estado)",
            2: "Evolui para estruturas simples, estáveis ou periódicas",
            3: "Comportamento caótico e aparentemente aleatório",
            4: "Estruturas complexas localizadas, potencialmente computação universal"
        }
        return descricoes.get(classe, "Comportamento não classificado")
    
    def classificar_multiplas_regras(self, regras: List[int], **kwargs) -> Dict[int, Dict]:
        """
        Classifica múltiplas regras.
        
        Args:
            regras: Lista de regras para classificar
            **kwargs: Argumentos passados para classificar_regra
            
        Returns:
            Dicionário mapeando regra para classificação
        """
        resultados = {}
        
        for regra in regras:
            try:
                resultados[regra] = self.classificar_regra(regra, **kwargs)
            except Exception as e:
                resultados[regra] = {
                    'regra': regra,
                    'erro': str(e),
                    'classe': None
                }
        
        return resultados
    
    def obter_estatisticas_classificacao(self, regras: List[int] = None) -> Dict:
        """
        Obtém estatísticas sobre a distribuição de classes.
        
        Args:
            regras: Lista de regras (se None, usa todas as 256)
            
        Returns:
            Estatísticas da classificação
        """
        if regras is None:
            regras = list(range(256))
        
        classificacoes = self.classificar_multiplas_regras(regras)
        
        # Contar por classe
        contadores = {1: 0, 2: 0, 3: 0, 4: 0, None: 0}
        
        for resultado in classificacoes.values():
            classe = resultado.get('classe')
            contadores[classe] = contadores.get(classe, 0) + 1
        
        total = len(regras)
        
        return {
            'total_regras': total,
            'por_classe': contadores,
            'percentuais': {
                classe: (count / total) * 100 
                for classe, count in contadores.items()
            },
            'regras_por_classe': {
                classe: [regra for regra, resultado in classificacoes.items() 
                        if resultado.get('classe') == classe]
                for classe in [1, 2, 3, 4]
            }
        }
