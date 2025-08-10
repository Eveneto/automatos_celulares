# Autômatos Celulares Elementares de Wolfram

Este projeto implementa os 256 autômatos celulares elementares descobertos por Stephen Wolfram, demonstrando como regras simples podem gerar padrões complexos e comportamentos emergentes.

## 📋 Sobre os Autômatos Celulares Elementares

Os autômatos celulares elementares são sistemas unidimensionais onde:
- Cada célula pode estar em um de dois estados: 0 (branco) ou 1 (preto)
- O estado de cada célula na próxima geração depende do seu estado atual e dos seus dois vizinhos
- Existem 2³ = 8 configurações possíveis de vizinhança
- Cada regra é definida por um número de 0 a 255 (2⁸ possibilidades)

## 🎯 Características Implementadas

- ✅ Implementação completa dos 256 autômatos elementares
- ✅ Visualização gráfica dos padrões gerados
- ✅ Análise de complexidade segundo classificação de Wolfram
- ✅ Detecção de padrões periódicos
- ✅ Exportação de imagens e dados
- ✅ Interface interativa para experimentação

## 🏗️ Estrutura do Projeto

```
src/
├── automato_elementar.py    # Classe principal do autômato
├── visualizador.py          # Sistema de visualização
├── analisador.py           # Análise de padrões e complexidade
├── classificador.py        # Classificação de Wolfram
└── utils.py                # Utilitários gerais

exemplos/
├── regras_famosas.py       # Exemplos de regras conhecidas
├── fractais.py             # Regras que geram fractais
└── experimentos.py         # Experimentos e análises

testes/
└── test_automato.py        # Testes unitários
```

## 🚀 Como Usar

```python
from src.automato_elementar import AutomatoElementar
from src.visualizador import Visualizador

# Criar autômato com regra 30 (caótica)
automato = AutomatoElementar(regra=30, tamanho=101)
automato.evoluir(geracoes=100)

# Visualizar
viz = Visualizador(automato)
viz.mostrar()
```

## 📊 Classificação de Wolfram

- **Classe I**: Evolui para estado homogêneo
- **Classe II**: Evolui para estruturas simples e periódicas
- **Classe III**: Comportamento caótico
- **Classe IV**: Estruturas complexas localizadas (computação universal)

## 🔬 Regras Famosas

- **Regra 30**: Caótica, usada em geradores de números aleatórios
- **Regra 90**: Gera triângulo de Sierpinski
- **Regra 110**: Computacionalmente universal
- **Regra 150**: Padrões fractais simples

## 📦 Dependências

```
numpy
matplotlib
pillow
jupyter
```

Instale com: `pip install -r requirements.txt`
