"""
Autômato Celular Elementar de Wolfram

Esta classe implementa um autômato celular unidimensional seguindo as regras
elementares de Wolfram (regras 0-255).
"""

import numpy as np
from typing import List, Tuple, Optional
import copy


class AutomatoElementar:
    """
    Implementa um autômato celular elementar de Wolfram.
    
    Os autômatos celulares elementares são sistemas unidimensionais onde cada
    célula pode estar em um de dois estados (0 ou 1) e evolui baseado em regras
    simples que dependem do estado da célula e de seus vizinhos.
    """
    
    def __init__(self, regra: int, tamanho: int = 101, condicao_contorno: str = 'circular'):
        """
        Inicializa o autômato celular elementar.
        
        Args:
            regra: Número da regra (0-255) que define o comportamento do autômato
            tamanho: Número de células na grade unidimensional
            condicao_contorno: Tipo de condição de contorno ('circular' ou 'fixo')
        """
        if not 0 <= regra <= 255:
            raise ValueError("Regra deve estar entre 0 e 255")
        
        self.regra = regra
        self.tamanho = tamanho
        self.condicao_contorno = condicao_contorno
        
        # Converter regra para tabela de lookup binária
        self.tabela_regra = self._criar_tabela_regra(regra)
        
        # Estado atual e histórico
        self.estado_atual = np.zeros(tamanho, dtype=int)
        self.historico = []
        self.geracao_atual = 0
        
        # Inicializar com uma única célula ativa no centro
        self.resetar()
    
    def _criar_tabela_regra(self, regra: int) -> dict:
        """
        Cria a tabela de lookup para a regra especificada.
        
        Args:
            regra: Número da regra (0-255)
            
        Returns:
            Dicionário mapeando configurações de vizinhança para novos estados
        """
        # Converter regra para binário (8 bits)
        regra_binaria = format(regra, '08b')
        
        # Configurações possíveis de vizinhança (esquerda, centro, direita)
        configuracoes = [
            (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
            (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0)
        ]
        
        # Mapear cada configuração para o bit correspondente da regra
        tabela = {}
        for i, config in enumerate(configuracoes):
            # Bit mais significativo corresponde à configuração (1,1,1)
            tabela[config] = int(regra_binaria[i])
        
        return tabela
    
    def resetar(self, estado_inicial: Optional[np.ndarray] = None):
        """
        Reseta o autômato para o estado inicial.
        
        Args:
            estado_inicial: Estado inicial customizado. Se None, usa célula central ativa.
        """
        if estado_inicial is not None:
            if len(estado_inicial) != self.tamanho:
                raise ValueError(f"Estado inicial deve ter {self.tamanho} elementos")
            self.estado_atual = np.array(estado_inicial, dtype=int)
        else:
            # Estado padrão: apenas célula central ativa
            self.estado_atual = np.zeros(self.tamanho, dtype=int)
            self.estado_atual[self.tamanho // 2] = 1
        
        self.historico = [self.estado_atual.copy()]
        self.geracao_atual = 0
    
    def _obter_vizinhanca(self, posicao: int) -> Tuple[int, int, int]:
        """
        Obtém a vizinhança de uma célula (esquerda, centro, direita).
        
        Args:
            posicao: Posição da célula
            
        Returns:
            Tupla com os estados (esquerda, centro, direita)
        """
        if self.condicao_contorno == 'circular':
            esquerda = self.estado_atual[(posicao - 1) % self.tamanho]
            direita = self.estado_atual[(posicao + 1) % self.tamanho]
        elif self.condicao_contorno == 'fixo':
            esquerda = self.estado_atual[posicao - 1] if posicao > 0 else 0
            direita = self.estado_atual[posicao + 1] if posicao < self.tamanho - 1 else 0
        else:
            raise ValueError("Condição de contorno deve ser 'circular' ou 'fixo'")
        
        centro = self.estado_atual[posicao]
        return (esquerda, centro, direita)
    
    def proximo_passo(self) -> np.ndarray:
        """
        Calcula o próximo estado do autômato.
        
        Returns:
            Novo estado do autômato
        """
        novo_estado = np.zeros(self.tamanho, dtype=int)
        
        for i in range(self.tamanho):
            vizinhanca = self._obter_vizinhanca(i)
            novo_estado[i] = self.tabela_regra[vizinhanca]
        
        return novo_estado
    
    def evoluir(self, geracoes: int = 1) -> List[np.ndarray]:
        """
        Evolui o autômato por um número especificado de gerações.
        
        Args:
            geracoes: Número de gerações para evoluir
            
        Returns:
            Lista com todos os estados (incluindo o inicial)
        """
        for _ in range(geracoes):
            self.estado_atual = self.proximo_passo()
            self.historico.append(self.estado_atual.copy())
            self.geracao_atual += 1
        
        return self.historico
    
    def obter_matriz_evolucao(self) -> np.ndarray:
        """
        Retorna a evolução completa como uma matriz 2D.
        
        Returns:
            Matriz onde cada linha representa uma geração
        """
        if not self.historico:
            return np.array([])
        
        return np.array(self.historico)
    
    def detectar_periodo(self, janela_busca: int = 20) -> Optional[int]:
        """
        Detecta se o autômato entrou em um ciclo periódico.
        
        Args:
            janela_busca: Tamanho máximo do período a ser detectado
            
        Returns:
            Tamanho do período detectado ou None se não periódico
        """
        if len(self.historico) < 2 * janela_busca:
            return None
        
        estado_atual = self.historico[-1]
        
        # Procurar por períodos de tamanho 1 até janela_busca
        for periodo in range(1, min(janela_busca + 1, len(self.historico) // 2)):
            # Verificar se os últimos 'periodo' estados se repetem
            repete = True
            for i in range(periodo):
                if not np.array_equal(self.historico[-(i+1)], self.historico[-(i+1+periodo)]):
                    repete = False
                    break
            
            if repete:
                return periodo
        
        return None
    
    def calcular_densidade(self) -> float:
        """
        Calcula a densidade de células ativas no estado atual.
        
        Returns:
            Proporção de células ativas (0.0 a 1.0)
        """
        return np.mean(self.estado_atual)
    
    def obter_estatisticas(self) -> dict:
        """
        Retorna estatísticas sobre a evolução do autômato.
        
        Returns:
            Dicionário com estatísticas diversas
        """
        if not self.historico:
            return {}
        
        matriz = self.obter_matriz_evolucao()
        densidades = [np.mean(estado) for estado in self.historico]
        
        return {
            'regra': self.regra,
            'geracoes': len(self.historico),
            'tamanho': self.tamanho,
            'densidade_inicial': densidades[0],
            'densidade_final': densidades[-1],
            'densidade_media': np.mean(densidades),
            'densidade_max': np.max(densidades),
            'densidade_min': np.min(densidades),
            'periodo_detectado': self.detectar_periodo(),
            'condicao_contorno': self.condicao_contorno
        }
    
    def __str__(self) -> str:
        """Representação em string do estado atual."""
        return ''.join(['█' if cell else '░' for cell in self.estado_atual])
    
    def __repr__(self) -> str:
        """Representação técnica do autômato."""
        return f"AutomatoElementar(regra={self.regra}, tamanho={self.tamanho}, geracao={self.geracao_atual})"
