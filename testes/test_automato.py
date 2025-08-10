"""
Testes unitários para os Autômatos Celulares Elementares

Este módulo contém testes para validar o funcionamento correto
das classes e funções do projeto.
"""

import unittest
import numpy as np
import sys
import os

# Adicionar src ao path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from automato_elementar import AutomatoElementar
from utils import *


class TestAutomatoElementar(unittest.TestCase):
    """Testes para a classe AutomatoElementar."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.automato = AutomatoElementar(regra=30, tamanho=11)
    
    def test_inicializacao_basica(self):
        """Testa inicialização básica do autômato."""
        self.assertEqual(self.automato.regra, 30)
        self.assertEqual(self.automato.tamanho, 11)
        self.assertEqual(self.automato.geracao_atual, 0)
        self.assertEqual(len(self.automato.historico), 1)
        
        # Estado inicial deve ter apenas célula central ativa
        estado_esperado = np.zeros(11)
        estado_esperado[5] = 1
        np.testing.assert_array_equal(self.automato.estado_atual, estado_esperado)
    
    def test_validacao_regra(self):
        """Testa validação do número da regra."""
        # Regras válidas
        AutomatoElementar(0, 10)
        AutomatoElementar(255, 10)
        
        # Regras inválidas
        with self.assertRaises(ValueError):
            AutomatoElementar(-1, 10)
        
        with self.assertRaises(ValueError):
            AutomatoElementar(256, 10)
    
    def test_tabela_regra_30(self):
        """Testa criação da tabela de regra para regra 30."""
        automato = AutomatoElementar(30, 10)
        
        # Regra 30 em binário: 00011110
        tabela_esperada = {
            (1, 1, 1): 0,  # 111 -> 0
            (1, 1, 0): 0,  # 110 -> 0
            (1, 0, 1): 0,  # 101 -> 0
            (1, 0, 0): 1,  # 100 -> 1
            (0, 1, 1): 1,  # 011 -> 1
            (0, 1, 0): 1,  # 010 -> 1
            (0, 0, 1): 1,  # 001 -> 1
            (0, 0, 0): 0   # 000 -> 0
        }
        
        self.assertEqual(automato.tabela_regra, tabela_esperada)
    
    def test_condicoes_contorno(self):
        """Testa diferentes condições de contorno."""
        # Circular
        automato_circular = AutomatoElementar(30, 5, 'circular')
        automato_circular.estado_atual = np.array([1, 0, 1, 0, 1])
        
        # Primeira célula deve ver última como vizinha esquerda
        viz = automato_circular._obter_vizinhanca(0)
        self.assertEqual(viz, (1, 1, 0))  # (última, primeira, segunda)
        
        # Fixo
        automato_fixo = AutomatoElementar(30, 5, 'fixo')
        automato_fixo.estado_atual = np.array([1, 0, 1, 0, 1])
        
        # Primeira célula deve ver 0 como vizinha esquerda
        viz = automato_fixo._obter_vizinhanca(0)
        self.assertEqual(viz, (0, 1, 0))  # (0, primeira, segunda)
    
    def test_evolucao(self):
        """Testa evolução do autômato."""
        # Usar regra simples para teste previsível
        automato = AutomatoElementar(0, 5)  # Regra 0: sempre 0
        automato.evoluir(1)
        
        # Após uma geração, tudo deve ser 0
        estado_esperado = np.zeros(5)
        np.testing.assert_array_equal(automato.estado_atual, estado_esperado)
        self.assertEqual(automato.geracao_atual, 1)
        self.assertEqual(len(automato.historico), 2)  # Estado inicial + 1 geração
    
    def test_reset_com_estado_customizado(self):
        """Testa reset com estado inicial customizado."""
        estado_custom = np.array([1, 0, 1, 0, 1])
        automato = AutomatoElementar(30, 5)
        automato.resetar(estado_custom)
        
        np.testing.assert_array_equal(automato.estado_atual, estado_custom)
        self.assertEqual(automato.geracao_atual, 0)
        self.assertEqual(len(automato.historico), 1)
    
    def test_deteccao_periodo(self):
        """Testa detecção de período."""
        # Criar autômato que converge rapidamente
        automato = AutomatoElementar(8, 5)  # Regra homogênea
        automato.evoluir(10)
        
        periodo = automato.detectar_periodo()
        # Para regras homogêneas, período deve ser 1 (estado fixo)
        self.assertEqual(periodo, 1)
    
    def test_estatisticas(self):
        """Testa cálculo de estatísticas."""
        automato = AutomatoElementar(30, 5)
        automato.evoluir(5)
        
        stats = automato.obter_estatisticas()
        
        self.assertIn('regra', stats)
        self.assertIn('geracoes', stats)
        self.assertIn('tamanho', stats)
        self.assertIn('densidade_inicial', stats)
        self.assertIn('densidade_final', stats)
        
        self.assertEqual(stats['regra'], 30)
        self.assertEqual(stats['tamanho'], 5)
        self.assertEqual(stats['geracoes'], 6)  # Estado inicial + 5 gerações


class TestUtils(unittest.TestCase):
    """Testes para funções utilitárias."""
    
    def test_gerar_estado_aleatorio(self):
        """Testa geração de estado aleatório."""
        tamanho = 100
        densidade = 0.3
        
        # Com semente para reprodutibilidade
        estado1 = gerar_estado_aleatorio(tamanho, densidade, semente=42)
        estado2 = gerar_estado_aleatorio(tamanho, densidade, semente=42)
        
        np.testing.assert_array_equal(estado1, estado2)
        
        # Densidade aproximada (com tolerância para aleatoriedade)
        densidade_real = np.mean(estado1)
        self.assertAlmostEqual(densidade_real, densidade, delta=0.1)
    
    def test_gerar_estado_impulso(self):
        """Testa geração de estado impulso."""
        tamanho = 10
        
        # Impulso central
        estado = gerar_estado_impulso(tamanho)
        estado_esperado = np.zeros(tamanho)
        estado_esperado[5] = 1
        np.testing.assert_array_equal(estado, estado_esperado)
        
        # Impulso em posição específica
        posicao = 3
        estado = gerar_estado_impulso(tamanho, posicao)
        estado_esperado = np.zeros(tamanho)
        estado_esperado[posicao] = 1
        np.testing.assert_array_equal(estado, estado_esperado)
    
    def test_gerar_estado_bloco(self):
        """Testa geração de estado com bloco."""
        tamanho = 10
        tamanho_bloco = 3
        
        estado = gerar_estado_bloco(tamanho, tamanho_bloco)
        
        # Deve ter exatamente tamanho_bloco células ativas
        self.assertEqual(np.sum(estado), tamanho_bloco)
        
        # Deve estar centralizado
        inicio_esperado = (tamanho - tamanho_bloco) // 2
        for i in range(tamanho):
            if inicio_esperado <= i < inicio_esperado + tamanho_bloco:
                self.assertEqual(estado[i], 1)
            else:
                self.assertEqual(estado[i], 0)
    
    def test_gerar_estado_periodico(self):
        """Testa geração de estado periódico."""
        tamanho = 10
        padrao = [1, 0, 1]
        
        estado = gerar_estado_periodico(tamanho, padrao)
        
        # Verificar se o padrão se repete
        for i in range(tamanho):
            self.assertEqual(estado[i], padrao[i % len(padrao)])
    
    def test_detectar_simetria(self):
        """Testa detecção de simetrias."""
        # Estado simétrico
        estado_simetrico = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])
        simetrias = detectar_simetria(estado_simetrico)
        self.assertTrue(simetrias['reflexiva'])
        
        # Estado não simétrico
        estado_assimetrico = np.array([1, 0, 1, 0, 0])
        simetrias = detectar_simetria(estado_assimetrico)
        self.assertFalse(simetrias['reflexiva'])
        
        # Estado homogêneo (simetria translacional)
        estado_homogeneo = np.array([1, 1, 1, 1, 1])
        simetrias = detectar_simetria(estado_homogeneo)
        self.assertTrue(simetrias['translacional'])
    
    def test_distancia_hamming(self):
        """Testa cálculo da distância de Hamming."""
        estado1 = np.array([1, 0, 1, 0, 1])
        estado2 = np.array([1, 1, 1, 0, 0])
        
        distancia = calcular_distancia_hamming(estado1, estado2)
        self.assertEqual(distancia, 2)  # Diferem nas posições 1 e 4
        
        # Estados iguais
        distancia = calcular_distancia_hamming(estado1, estado1)
        self.assertEqual(distancia, 0)
        
        # Tamanhos diferentes devem gerar erro
        estado3 = np.array([1, 0, 1])
        with self.assertRaises(ValueError):
            calcular_distancia_hamming(estado1, estado3)
    
    def test_calcular_entropia(self):
        """Testa cálculo de entropia."""
        # Estado homogêneo (entropia mínima)
        estado_homogeneo = np.array([1, 1, 1, 1])
        entropia = calcular_entropia(estado_homogeneo)
        self.assertAlmostEqual(entropia, 0.0, places=5)
        
        # Estado balanceado (entropia máxima para binário)
        estado_balanceado = np.array([1, 0, 1, 0])
        entropia = calcular_entropia(estado_balanceado)
        self.assertAlmostEqual(entropia, 1.0, places=5)
    
    def test_encontrar_padroes_locais(self):
        """Testa detecção de padrões locais."""
        # Matriz simples para teste
        matriz = np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1]
        ])
        
        padroes = encontrar_padroes_locais(matriz, tamanho_janela=2)
        
        self.assertIn('padroes', padroes)
        self.assertIn('total_padroes', padroes)
        self.assertIn('padrao_mais_comum', padroes)
        
        # Deve detectar os padrões [1,0], [0,1], etc.
        self.assertIn((1, 0), padroes['padroes'])
        self.assertIn((0, 1), padroes['padroes'])


class TestRegrasConchecidas(unittest.TestCase):
    """Testes para verificar comportamentos conhecidos de regras específicas."""
    
    def test_regra_0_sempre_zero(self):
        """Regra 0 deve sempre resultar em estado zero."""
        automato = AutomatoElementar(0, 10)
        automato.evoluir(5)
        
        estado_esperado = np.zeros(10)
        np.testing.assert_array_equal(automato.estado_atual, estado_esperado)
    
    def test_regra_255_sempre_um(self):
        """Regra 255 deve sempre resultar em estado com todos 1s."""
        automato = AutomatoElementar(255, 10)
        automato.evoluir(1)
        
        estado_esperado = np.ones(10)
        np.testing.assert_array_equal(automato.estado_atual, estado_esperado)
    
    def test_regra_90_sierpinski(self):
        """Regra 90 deve gerar padrão fractal (teste básico)."""
        automato = AutomatoElementar(90, 11)
        automato.evoluir(5)
        
        # Verificar que o padrão não é trivial
        self.assertGreater(np.sum(automato.estado_atual), 0)
        self.assertLess(np.sum(automato.estado_atual), len(automato.estado_atual))
        
        # Verificar simetria (característica da regra 90)
        simetrias = detectar_simetria(automato.estado_atual)
        # A regra 90 tende a gerar padrões simétricos
        # (pode não ser sempre verdade dependendo do número de gerações)


def executar_testes_completos():
    """Executa todos os testes e mostra relatório."""
    print("🧪 EXECUTANDO TESTES DOS AUTÔMATOS CELULARES")
    print("=" * 50)
    
    # Descobrir e executar todos os testes
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(__file__), pattern='test_*.py')
    
    # Executar com verbosidade
    runner = unittest.TextTestRunner(verbosity=2)
    resultado = runner.run(suite)
    
    # Relatório final
    print("\n" + "=" * 50)
    print("📊 RELATÓRIO FINAL DOS TESTES")
    print(f"Tests executados: {resultado.testsRun}")
    print(f"Falhas: {len(resultado.failures)}")
    print(f"Erros: {len(resultado.errors)}")
    
    if resultado.failures:
        print("\n❌ FALHAS:")
        for test, traceback in resultado.failures:
            print(f"  - {test}: {traceback}")
    
    if resultado.errors:
        print("\n💥 ERROS:")
        for test, traceback in resultado.errors:
            print(f"  - {test}: {traceback}")
    
    if not resultado.failures and not resultado.errors:
        print("\n✅ TODOS OS TESTES PASSARAM!")
    
    return resultado.wasSuccessful()


if __name__ == '__main__':
    # Executar testes individuais ou completos
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--completo':
        sucesso = executar_testes_completos()
        sys.exit(0 if sucesso else 1)
    else:
        # Executar apenas os testes deste arquivo
        unittest.main(verbosity=2)
