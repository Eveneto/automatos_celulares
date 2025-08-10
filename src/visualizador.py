"""
Sistema de Visualização para Autômatos Celulares Elementares

Este módulo fornece ferramentas para visualizar a evolução dos autômatos
celulares elementares usando matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from typing import Optional, Tuple, List
import os
from datetime import datetime

try:
    from .automato_elementar import AutomatoElementar
except ImportError:
    from automato_elementar import AutomatoElementar


class Visualizador:
    """
    Classe para visualização de autômatos celulares elementares.
    """
    
    def __init__(self, automato: AutomatoElementar):
        """
        Inicializa o visualizador.
        
        Args:
            automato: Instância do autômato celular a ser visualizado
        """
        self.automato = automato
        
        # Configurações de cores
        self.cores = {
            'classico': ['white', 'black'],
            'azul': ['lightblue', 'darkblue'],
            'verde': ['lightgreen', 'darkgreen'],
            'vermelho': ['pink', 'darkred'],
            'roxo': ['lavender', 'purple']
        }
        self.esquema_cor_atual = 'classico'
    
    def definir_esquema_cor(self, esquema: str):
        """
        Define o esquema de cores para visualização.
        
        Args:
            esquema: Nome do esquema ('classico', 'azul', 'verde', 'vermelho', 'roxo')
        """
        if esquema in self.cores:
            self.esquema_cor_atual = esquema
        else:
            raise ValueError(f"Esquema '{esquema}' não disponível. Use: {list(self.cores.keys())}")
    
    def mostrar_evolucao(self, figsize: Tuple[int, int] = (12, 8), salvar: bool = False, 
                        nome_arquivo: Optional[str] = None) -> plt.Figure:
        """
        Mostra a evolução completa do autômato como uma imagem 2D.
        
        Args:
            figsize: Tamanho da figura (largura, altura)
            salvar: Se True, salva a imagem
            nome_arquivo: Nome do arquivo para salvar (opcional)
            
        Returns:
            Figura matplotlib
        """
        matriz = self.automato.obter_matriz_evolucao()
        
        if matriz.size == 0:
            raise ValueError("Autômato não foi evoluído ainda")
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Configurar colormap
        cmap = ListedColormap(self.cores[self.esquema_cor_atual])
        
        # Plotar matriz
        im = ax.imshow(matriz, cmap=cmap, interpolation='nearest', aspect='auto')
        
        # Configurar título e labels
        stats = self.automato.obter_estatisticas()
        titulo = f"Regra {self.automato.regra} - {len(matriz)} gerações"
        
        if stats.get('periodo_detectado'):
            titulo += f" (Período: {stats['periodo_detectado']})"
        
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.set_xlabel('Posição na Grade', fontsize=12)
        ax.set_ylabel('Geração', fontsize=12)
        
        # Inverter eixo Y para mostrar evolução de cima para baixo
        ax.invert_yaxis()
        
        # Adicionar barra de cores
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Estado da Célula', fontsize=10)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Inativo (0)', 'Ativo (1)'])
        
        plt.tight_layout()
        
        # Salvar se solicitado
        if salvar:
            if nome_arquivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nome_arquivo = f"regra_{self.automato.regra}_{timestamp}.png"
            
            os.makedirs('imagens', exist_ok=True)
            caminho_completo = os.path.join('imagens', nome_arquivo)
            plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
            print(f"Imagem salva em: {caminho_completo}")
        
        return fig
    
    def mostrar_estado_atual(self, figsize: Tuple[int, int] = (15, 3)) -> plt.Figure:
        """
        Mostra apenas o estado atual do autômato.
        
        Args:
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Cores para células ativas e inativas
        cores = [self.cores[self.esquema_cor_atual][estado] for estado in self.automato.estado_atual]
        
        # Criar gráfico de barras
        posicoes = range(len(self.automato.estado_atual))
        ax.bar(posicoes, [1] * len(posicoes), color=cores, width=1.0, edgecolor='none')
        
        ax.set_title(f"Estado Atual - Regra {self.automato.regra} (Geração {self.automato.geracao_atual})")
        ax.set_xlabel('Posição')
        ax.set_ylabel('Estado')
        ax.set_ylim(0, 1.2)
        ax.set_xlim(-0.5, len(posicoes) - 0.5)
        
        # Remover ticks do eixo Y
        ax.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def criar_animacao(self, intervalo: int = 100, salvar: bool = False, 
                      nome_arquivo: Optional[str] = None) -> animation.FuncAnimation:
        """
        Cria uma animação da evolução do autômato.
        
        Args:
            intervalo: Intervalo entre frames em milissegundos
            salvar: Se True, salva a animação como GIF
            nome_arquivo: Nome do arquivo para salvar
            
        Returns:
            Objeto de animação matplotlib
        """
        if not self.automato.historico:
            raise ValueError("Autômato não foi evoluído ainda")
        
        fig, ax = plt.subplots(figsize=(15, 3))
        
        # Configuração inicial
        posicoes = range(self.automato.tamanho)
        barras = ax.bar(posicoes, [1] * self.automato.tamanho, width=1.0, edgecolor='none')
        
        ax.set_title(f"Evolução - Regra {self.automato.regra}")
        ax.set_xlabel('Posição')
        ax.set_ylim(0, 1.2)
        ax.set_xlim(-0.5, self.automato.tamanho - 0.5)
        ax.set_yticks([])
        
        # Texto para mostrar geração atual
        texto_geracao = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                               fontsize=12, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def atualizar_frame(frame):
            """Atualiza um frame da animação."""
            estado = self.automato.historico[frame]
            cores = [self.cores[self.esquema_cor_atual][cell] for cell in estado]
            
            for barra, cor in zip(barras, cores):
                barra.set_color(cor)
            
            texto_geracao.set_text(f'Geração: {frame}')
            return barras + [texto_geracao]
        
        # Criar animação
        anim = animation.FuncAnimation(
            fig, atualizar_frame, frames=len(self.automato.historico),
            interval=intervalo, blit=True, repeat=True
        )
        
        # Salvar se solicitado
        if salvar:
            if nome_arquivo is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nome_arquivo = f"animacao_regra_{self.automato.regra}_{timestamp}.gif"
            
            os.makedirs('animacoes', exist_ok=True)
            caminho_completo = os.path.join('animacoes', nome_arquivo)
            anim.save(caminho_completo, writer='pillow', fps=10)
            print(f"Animação salva em: {caminho_completo}")
        
        return anim
    
    def comparar_regras(self, regras: List[int], geracoes: int = 50, 
                       figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Compara múltiplas regras lado a lado.
        
        Args:
            regras: Lista de números de regras para comparar
            geracoes: Número de gerações para evoluir cada regra
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib com comparação
        """
        n_regras = len(regras)
        fig, axes = plt.subplots(1, n_regras, figsize=figsize, sharey=True)
        
        if n_regras == 1:
            axes = [axes]
        
        cmap = ListedColormap(self.cores[self.esquema_cor_atual])
        
        for i, regra in enumerate(regras):
            # Criar novo autômato para cada regra
            automato_temp = AutomatoElementar(regra, self.automato.tamanho)
            automato_temp.evoluir(geracoes)
            matriz = automato_temp.obter_matriz_evolucao()
            
            # Plotar
            im = axes[i].imshow(matriz, cmap=cmap, interpolation='nearest', aspect='auto')
            axes[i].set_title(f'Regra {regra}', fontweight='bold')
            axes[i].set_xlabel('Posição')
            
            if i == 0:
                axes[i].set_ylabel('Geração')
            
            # Inverter eixo Y
            axes[i].invert_yaxis()
        
        plt.suptitle('Comparação de Regras de Autômatos Celulares', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plotar_densidade_temporal(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plota a evolução da densidade de células ativas ao longo do tempo.
        
        Args:
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        if not self.automato.historico:
            raise ValueError("Autômato não foi evoluído ainda")
        
        # Calcular densidades
        densidades = [np.mean(estado) for estado in self.automato.historico]
        geracoes = range(len(densidades))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(geracoes, densidades, linewidth=2, color='darkblue', marker='o', markersize=3)
        ax.set_title(f'Evolução da Densidade - Regra {self.automato.regra}', fontweight='bold')
        ax.set_xlabel('Geração')
        ax.set_ylabel('Densidade (proporção de células ativas)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Adicionar linha de densidade média
        densidade_media = np.mean(densidades)
        ax.axhline(y=densidade_media, color='red', linestyle='--', alpha=0.7,
                  label=f'Densidade média: {densidade_media:.3f}')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def mostrar_regra_binaria(self, figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Mostra a representação visual da regra como tabela de lookup.
        
        Args:
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 1])
        
        # Configurações de vizinhança
        configuracoes = [
            (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
            (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0)
        ]
        
        # Plotar configurações de entrada
        for i, config in enumerate(configuracoes):
            for j, valor in enumerate(config):
                cor = self.cores[self.esquema_cor_atual][valor]
                ax1.add_patch(plt.Rectangle((i*3 + j, 0), 1, 1, facecolor=cor, edgecolor='black'))
        
        ax1.set_xlim(0, 24)
        ax1.set_ylim(0, 1)
        ax1.set_title('Configurações de Entrada (Esquerda, Centro, Direita)', fontweight='bold')
        ax1.set_xticks([1.5 + i*3 for i in range(8)])
        ax1.set_xticklabels([f'{sum(c)}' for c in configuracoes])
        ax1.set_yticks([])
        
        # Plotar saídas correspondentes
        regra_binaria = format(self.automato.regra, '08b')
        for i, bit in enumerate(regra_binaria):
            cor = self.cores[self.esquema_cor_atual][int(bit)]
            ax2.add_patch(plt.Rectangle((i*3 + 1, 0), 1, 1, facecolor=cor, edgecolor='black'))
        
        ax2.set_xlim(0, 24)
        ax2.set_ylim(0, 1)
        ax2.set_title(f'Saídas da Regra {self.automato.regra} ({regra_binaria})', fontweight='bold')
        ax2.set_xticks([1.5 + i*3 for i in range(8)])
        ax2.set_xticklabels([bit for bit in regra_binaria])
        ax2.set_yticks([])
        
        plt.tight_layout()
        return fig
