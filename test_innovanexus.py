"""
Testes unitários para o InnovaNexus.
Cobre:
1. Fluxo de Análise Sob Demanda vs Automático (construção de prompts e validações).
2. Módulo de Ciência de Dados: Regressão Linear, Projeções Temporais e Bivariadas.
3. Cálculos Estatísticos Avançados: Coeficiente de Variação, Quartis, IQR, Outliers.
4. Casos de borda: dados ausentes (NaN/None/Inf), colunas não numéricas,
   variância zero, amostras insuficientes (< 3 registros), média zero.
"""

import unittest
import numpy as np
import pandas as pd

from data_science import (
    formatar_valor,
    calcular_estatisticas_resumo,
    classificar_quartis_outliers,
    treinar_regressao_linear,
    executar_previsao_temporal,
    executar_regressao_bivariada,
)
from app import construir_prompt_insights


class TestAnaliseSobDemanda(unittest.TestCase):
    """Testes para o fluxo de análise automática e sob demanda."""

    def setUp(self):
        self.info_str = "Colunas: data (datetime), valor (float), categoria (object)"
        self.describe_str = "valor: mean=150, std=30, min=100, max=200"
        self.head_str = "2024-01-01, 100.0, CatA\n2024-02-01, 150.0, CatB"

    def test_prompt_automatico_padrao(self):
        """Verifica se o modo automático mantém o prompt executivo padrão."""
        prompt = construir_prompt_insights(
            self.info_str, self.describe_str, self.head_str,
            tipo_analise="Automático (Geral)", pergunta_usuario=""
        )
        self.assertIn("InnovaNexus - Relatório Executivo de Dados", prompt)
        self.assertIn("Objetivo Estratégico", prompt)
        self.assertIn("Indicadores-Chave de Desempenho (KPIs)", prompt)
        self.assertIn("Insights Estratégicos da Operação", prompt)
        self.assertIn("Ações Prescritivas para a Gestão", prompt)
        self.assertNotIn("ANÁLISE SOB DEMANDA", prompt)

    def test_prompt_sob_demanda_com_tipo_analise(self):
        """Verifica customização com tipo de análise específico."""
        prompt = construir_prompt_insights(
            self.info_str, self.describe_str, self.head_str,
            tipo_analise="Diagnóstico Operacional & Gargalos",
            pergunta_usuario=""
        )
        self.assertIn("ANÁLISE SOB DEMANDA", prompt)
        self.assertIn("Diagnóstico Operacional & Gargalos", prompt)
        self.assertIn("Resposta Direta & Diagnóstico da Pergunta Solicitada", prompt)
        self.assertIn("Evidências dos Dados & Indicadores Relevantes", prompt)

    def test_prompt_sob_demanda_com_pergunta(self):
        """Verifica se a pergunta específica do usuário é incluída no prompt."""
        pergunta = "Qual categoria apresenta o maior atraso e qual o custo estimado?"
        prompt = construir_prompt_insights(
            self.info_str, self.describe_str, self.head_str,
            tipo_analise="Automático (Geral)",
            pergunta_usuario=pergunta
        )
        self.assertIn("ANÁLISE SOB DEMANDA", prompt)
        self.assertIn(pergunta, prompt)
        self.assertIn("Resposta Direta & Diagnóstico da Pergunta Solicitada", prompt)

    def test_prompt_sob_demanda_completo(self):
        """Verifica tipo de análise + pergunta do usuário combinados."""
        pergunta = "Como otimizar os custos da rota Sudeste?"
        tipo = "Financeiro & Rentabilidade"
        prompt = construir_prompt_insights(
            self.info_str, self.describe_str, self.head_str,
            tipo_analise=tipo,
            pergunta_usuario=pergunta
        )
        self.assertIn(tipo, prompt)
        self.assertIn(pergunta, prompt)
        self.assertIn("--- DADOS PARA ANÁLISE ---", prompt)


class TestRegressaoLinear(unittest.TestCase):
    """Testes dos cálculos de regressão linear."""

    def test_regressao_linear_perfeita(self):
        """Relação linear y = 3x + 5 deve ter R² = 1.0."""
        X = np.array([1, 2, 3, 4, 5])
        y = np.array([8, 11, 14, 17, 20])  # 3x + 5
        res = treinar_regressao_linear(X, y)

        self.assertAlmostEqual(res["r2"], 1.0, places=4)
        self.assertAlmostEqual(res["coef"], 3.0, places=4)
        self.assertAlmostEqual(res["intercept"], 5.0, places=4)
        self.assertAlmostEqual(res["mae"], 0.0, places=4)
        self.assertAlmostEqual(res["rmse"], 0.0, places=4)

    def test_regressao_com_dados_ausentes(self):
        """Valores NaN e Inf devem ser ignorados, ajustando sobre os válidos."""
        X = np.array([1, 2, np.nan, 4, 5, np.inf])
        y = np.array([8, 11, 14, 17, 20, 23])
        res = treinar_regressao_linear(X, y)

        # Devem restar 4 pontos válidos (1, 2, 4, 5)
        self.assertEqual(res["n_obs"], 4)
        self.assertAlmostEqual(res["coef"], 3.0, places=4)
        self.assertAlmostEqual(res["r2"], 1.0, places=4)

    def test_regressao_amostra_insuficiente(self):
        """Menos de 3 observações válidas deve levantar ValueError."""
        X = np.array([1, 2])
        y = np.array([10, 20])
        with self.assertRaises(ValueError) as ctx:
            treinar_regressao_linear(X, y)
        self.assertIn("insuficientes", str(ctx.exception).lower())

    def test_regressao_variancia_zero_em_x(self):
        """Variável X com valores todos iguais deve levantar erro amigável."""
        X = np.array([5, 5, 5, 5])
        y = np.array([10, 20, 30, 40])
        with self.assertRaises(ValueError) as ctx:
            treinar_regressao_linear(X, y)
        self.assertIn("variância zero", str(ctx.exception).lower())

    def test_regressao_dimensoes_incompativeis(self):
        """X e y com comprimentos distintos devem levantar ValueError."""
        X = np.array([1, 2, 3])
        y = np.array([10, 20])
        with self.assertRaises(ValueError):
            treinar_regressao_linear(X, y)


class TestPrevisaoTemporal(unittest.TestCase):
    """Testes de projeção temporal."""

    def setUp(self):
        # Série de 6 meses com faturamento crescente
        self.df = pd.DataFrame({
            "data": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "faturamento": [10000, 12000, 14000, 16000, 18000, 20000],
            "quantidade": [100, 120, 135, 160, 175, 200]
        })

    def test_previsao_temporal_sucesso(self):
        """Testa geração de 3 períodos futuros."""
        res = executar_previsao_temporal(
            self.df, col_data="data", col_valor="faturamento",
            n_periodos=3, freq="M", agregacao="sum"
        )
        self.assertEqual(len(res["df_previsoes"]), 3)
        self.assertGreater(res["r2"], 0.99)
        self.assertAlmostEqual(res["coef"], 2000.0, places=1)
        # Próximo mês (mês índice 6) deve ser 22000
        primeira_prev = res["df_previsoes"]["valor_previsto"].iloc[0]
        self.assertAlmostEqual(primeira_prev, 22000.0, places=1)
        self.assertGreater(res["total_previsto"], 0)

    def test_previsao_temporal_dados_nulos_e_sujeira(self):
        """Testa robustez com NaNs e datas inválidas."""
        df_sujo = self.df.copy()
        df_sujo.loc[len(df_sujo)] = ["DATA_INVALIDA", 99999, 50]
        df_sujo.loc[len(df_sujo)] = [pd.NaT, np.nan, 0]

        res = executar_previsao_temporal(
            df_sujo, col_data="data", col_valor="faturamento",
            n_periodos=2, freq="M"
        )
        self.assertEqual(len(res["df_previsoes"]), 2)

    def test_previsao_temporal_periodos_insuficientes(self):
        """Menos de 3 meses deve levantar ValueError."""
        df_curto = self.df.head(2)
        with self.assertRaises(ValueError) as ctx:
            executar_previsao_temporal(df_curto, col_data="data", col_valor="faturamento")
        self.assertIn("insuficientes", str(ctx.exception).lower())


class TestRegressaoBivariada(unittest.TestCase):
    """Testes de regressão bivariada (X -> Y)."""

    def test_regressao_bivariada_sucesso(self):
        df = pd.DataFrame({
            "preco": [10.0, 20.0, 30.0, 40.0, 50.0],
            "vendas": [100.0, 200.0, 300.0, 400.0, 500.0],
            "categoria": ["A", "B", "C", "D", "E"]
        })
        res = executar_regressao_bivariada(df, "preco", "vendas")
        self.assertAlmostEqual(res["r2"], 1.0, places=4)
        self.assertAlmostEqual(res["coef"], 10.0, places=4)
        self.assertEqual(res["n_obs"], 5)

    def test_regressao_bivariada_coluna_nao_numerica(self):
        """Coluna inteiramente em texto/não numérica deve levantar ValueError."""
        df = pd.DataFrame({
            "texto": ["abc", "def", "ghi"],
            "vendas": [100.0, 200.0, 300.0]
        })
        with self.assertRaises(ValueError) as ctx:
            executar_regressao_bivariada(df, "texto", "vendas")
        self.assertIn("insuficientes", str(ctx.exception).lower())


class TestEstatisticasAvancadas(unittest.TestCase):
    """Testes de métricas estatísticas: CV, IQR, quartis, outliers e formatação."""

    def test_calculo_estatisticas_resumo(self):
        serie = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        res = calcular_estatisticas_resumo(serie)

        self.assertEqual(res["contagem"], 10)
        self.assertEqual(res["media"], 55.0)
        self.assertEqual(res["mediana"], 55.0)
        self.assertGreater(res["desvio_padrao"], 0)
        self.assertGreater(res["coef_variacao"], 0)
        self.assertEqual(res["iqr"], res["q3"] - res["q1"])

    def test_estatisticas_com_media_zero(self):
        """Média zero não deve causar ZeroDivisionError no Coeficiente de Variação."""
        serie = pd.Series([-10, 0, 10])
        res = calcular_estatisticas_resumo(serie)
        self.assertEqual(res["media"], 0.0)
        self.assertEqual(res["coef_variacao"], 0.0)

    def test_estatisticas_com_nulos_e_texto(self):
        """Valores de texto e NaNs devem ser ignorados."""
        serie = pd.Series([10, np.nan, "invalido", 20, 30])
        res = calcular_estatisticas_resumo(serie)
        self.assertEqual(res["contagem"], 3)
        self.assertEqual(res["media"], 20.0)

    def test_classificar_quartis_outliers(self):
        # Série com outlier evidente
        serie = pd.Series([10, 12, 11, 13, 12, 10, 11, 1000])
        classif = classificar_quartis_outliers(serie)
        self.assertIn("Outlier Superior", classif.values)

    def test_formatar_valor(self):
        self.assertEqual(formatar_valor(1500000), "R$ 1,50M")
        self.assertEqual(formatar_valor(2500), "R$ 2.500")
        self.assertEqual(formatar_valor(45.5), "R$ 45,50")
        self.assertEqual(formatar_valor(None), "N/A")
        self.assertEqual(formatar_valor(np.nan), "N/A")


if __name__ == "__main__":
    unittest.main()
