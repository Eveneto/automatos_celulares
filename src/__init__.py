"""
Autômatos Celulares Elementares de Wolfram

Este pacote implementa um sistema completo para simulação e análise
dos 256 autômatos celulares elementares descobertos por Stephen Wolfram.

Módulos principais:
- automato_elementar: Classe principal AutomatoElementar
- visualizador: Sistema de visualização com matplotlib
- classificador: Classificação de Wolfram em 4 classes
- utils: Funções utilitárias e ferramentas auxiliares

Exemplo de uso básico:
    from src.automato_elementar import AutomatoElementar
    from src.visualizador import Visualizador
    
    # Criar autômato com regra 30
    automato = AutomatoElementar(regra=30, tamanho=101)
    automato.evoluir(100)
    
    # Visualizar
    viz = Visualizador(automato)
    viz.mostrar_evolucao()
"""

__version__ = "1.0.0"
__author__ = "Projeto Autômatos Celulares"

# Importações principais para facilitar o uso
from .automato_elementar import AutomatoElementar
from .visualizador import Visualizador
from .classificador import ClassificadorWolfram

__all__ = [
    'AutomatoElementar',
    'Visualizador', 
    'ClassificadorWolfram'
]
