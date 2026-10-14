"""
Módulo de Ciência de Dados e Regressão para o InnovaNexus.
Integra recursos de regressão linear (temporal e bivariada), cálculos
estatísticos avançados (quartis, IQR, outliers, coeficiente de variação)
e projeções futuras, com tratamento robusto de erros e casos de borda.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def eh_coluna_monetaria(nome_coluna):
    """
    Identifica de forma inteligente se uma coluna representa valores monetários (moeda/R$).
    Evita falsos-positivos em variáveis de quantidade, tempo, idade, scores, etc.
    """
    if not nome_coluna or not isinstance(nome_coluna, str):
        return False

    col = nome_coluna.lower().strip()

    # Termos expressamente monetários (alta prioridade)
    termos_monetarios_diretos = [
        'nota_fiscal', 'faturamento', 'receita', 'despesa', 'lucro',
        'salario', 'salário', 'ticket', 'total_pago', 'frete',
        'orcamento', 'orçamento', 'dinheiro', 'r$', 'price', 'cost',
        'revenue', 'fee', 'salary', 'budget', 'sales', 'spend'
    ]
    if any(m in col for m in termos_monetarios_diretos):
        return True

    # Termos que expressamente indicam contagem, tempo, idade ou índices
    termos_nao_monetarios = [
        'quantidade', 'qtd', 'volume', 'count', 'numero', 'número', 'num_', '_num',
        'horas', 'hora', 'tempo', 'minutos', 'segundos', 'dias',
        'idade', 'age', 'ano', 'mes', 'mês', 'dia',
        'score', 'pontuacao', 'pontuação', 'indice', 'índice', 'ansiedade',
        'taxa', 'rate', 'pct', 'percent', '%', 'foco', 'sono',
        'grau', 'aluno', 'id', 'codigo', 'código', 'cep', 'periodo', 'seq'
    ]
    if any(t in col for t in termos_nao_monetarios):
        return False

    # Termos gerais monetários (como 'valor' ou 'preco')
    if any(m in col for m in ['valor', 'preco', 'preço', 'custo']):
        return True

    return False


def obter_prefixo_coluna(nome_coluna):
    """Retorna 'R$ ' se a coluna for monetária, ou '' caso contrário."""
    return "R$ " if eh_coluna_monetaria(nome_coluna) else ""


def formatar_valor(valor, prefixo=None, nome_coluna=None):
    """
    Formata valores monetários ou numéricos no padrão brasileiro.
    Se nome_coluna for informado e prefixo for None, detecta automaticamente se é monetário ('R$ ') ou numérico comum ('').
    Se nem prefixo nem nome_coluna forem fornecidos, o padrão é sem prefixo ('') para evitar rotular dados gerais como R$.
    """
    if valor is None or pd.isna(valor):
        return "N/A"
    try:
        val = float(valor)
    except (ValueError, TypeError):
        return str(valor)

    if prefixo is None:
        prefixo = obter_prefixo_coluna(nome_coluna) if nome_coluna else ""

    sinal = "-" if val < 0 else ""
    val_abs = abs(val)

    if val_abs >= 1_000_000:
        texto = f"{prefixo}{sinal}{val_abs / 1_000_000:,.2f}M"
        return texto.replace(",", "X").replace(".", ",").replace("X", ".")
    elif val_abs >= 1_000:
        if val_abs.is_integer() and not prefixo:
            return f"{sinal}{int(val_abs):,}".replace(",", ".")
        texto = f"{prefixo}{sinal}{val_abs:,.2f}" if (val_abs % 1 != 0) else f"{prefixo}{sinal}{val_abs:,.0f}"
        return texto.replace(",", "X").replace(".", ",").replace("X", ".")
    else:
        # Se for inteiro exato e sem prefixo monetário (ex: idade 16), formata como inteiro
        if val_abs.is_integer() and not prefixo:
            return f"{sinal}{int(val_abs)}"
        texto = f"{prefixo}{sinal}{val_abs:.2f}"
        return texto.replace(".", ",")


def calcular_estatisticas_resumo(serie):
    """
    Calcula métricas estatísticas detalhadas de uma série numérica:
    média, desvio padrão, coeficiente de variação, mediana, quartis,
    IQR e identificação de outliers (percentil 95 e critério IQR).
    """
    s_clean = pd.to_numeric(serie, errors="coerce").dropna()
    n = len(s_clean)
    if n == 0:
        raise ValueError("A coluna selecionada não contém valores numéricos válidos.")

    media = float(s_clean.mean())
    desvio = float(s_clean.std(ddof=1)) if n > 1 else 0.0
    coef_var = float((desvio / media * 100)) if media != 0 else 0.0
    mediana = float(s_clean.median())
    minimo = float(s_clean.min())
    maximo = float(s_clean.max())

    q1 = float(s_clean.quantile(0.25))
    q2 = mediana
    q3 = float(s_clean.quantile(0.75))
    iqr = q3 - q1

    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr

    p95 = float(s_clean.quantile(0.95))
    outliers_p95_count = int((s_clean > p95).sum())
    outliers_p95_pct = float((outliers_p95_count / n) * 100) if n > 0 else 0.0

    outliers_iqr_count = int(((s_clean < limite_inf) | (s_clean > limite_sup)).sum())
    outliers_iqr_pct = float((outliers_iqr_count / n) * 100) if n > 0 else 0.0

    return {
        "contagem": n,
        "media": media,
        "desvio_padrao": desvio,
        "coef_variacao": coef_var,
        "mediana": mediana,
        "minimo": minimo,
        "maximo": maximo,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": limite_inf,
        "limite_superior": limite_sup,
        "p95": p95,
        "outliers_p95_count": outliers_p95_count,
        "outliers_p95_pct": outliers_p95_pct,
        "outliers_iqr_count": outliers_iqr_count,
        "outliers_iqr_pct": outliers_iqr_pct,
    }


def classificar_quartis_outliers(serie):
    """
    Classifica cada elemento da série nas faixas de quartis e outliers (critério IQR).
    Retorna uma Series de categorias com a respectiva classificação.
    """
    s_clean = pd.to_numeric(serie, errors="coerce").dropna()
    if len(s_clean) == 0:
        return pd.Series(dtype=str)

    q1 = s_clean.quantile(0.25)
    q2 = s_clean.median()
    q3 = s_clean.quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr

    def _classificar(v):
        if v < limite_inf:
            return "Outlier Inferior"
        elif v <= q1:
            return "Q1 (25%)"
        elif v <= q2:
            return "Q2 (50%)"
        elif v <= q3:
            return "Q3 (75%)"
        elif v <= limite_sup:
            return "Q4 (100%)"
        else:
            return "Outlier Superior"

    return s_clean.apply(_classificar)


def treinar_regressao_linear(X, y):
    """
    Ajusta modelo LinearRegression sobre X e y com validação de dados ausentes,
    tamanho mínimo amostral e variância zero.

    Retorna:
        dict com modelo, métricas (r2, mae, rmse), coeficientes e arrays limpos.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    if len(X_arr) != len(y_arr):
        raise ValueError(
            f"Dimensões incompatíveis: X tem {len(X_arr)} itens e y tem {len(y_arr)} itens."
        )

    # Filtragem de NaNs e infinitos
    mask_valida = ~(np.isnan(X_arr).any(axis=1) | np.isinf(X_arr).any(axis=1) |
                    np.isnan(y_arr) | np.isinf(y_arr))

    X_clean = X_arr[mask_valida]
    y_clean = y_arr[mask_valida]

    if len(X_clean) < 3:
        raise ValueError(
            "Dados insuficientes para regressão linear. São necessárias pelo menos 3 observações válidas."
        )

    # Validação de variância zero na variável preditora
    if np.all(X_clean == X_clean[0]):
        raise ValueError(
            "A variável preditora (X) possui variância zero (todos os valores são idênticos). "
            "Não é possível traçar uma reta de regressão."
        )

    modelo = LinearRegression()
    modelo.fit(X_clean, y_clean)
    y_pred = modelo.predict(X_clean)

    r2 = float(r2_score(y_clean, y_pred))
    mae = float(mean_absolute_error(y_clean, y_pred))
    mse = float(mean_squared_error(y_clean, y_pred))
    rmse = float(np.sqrt(mse))

    coef = float(modelo.coef_[0]) if hasattr(modelo.coef_, "__iter__") else float(modelo.coef_)
    intercept = float(modelo.intercept_)

    return {
        "modelo": modelo,
        "y_pred": y_pred,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "coef": coef,
        "intercept": intercept,
        "X_clean": X_clean,
        "y_clean": y_clean,
        "n_obs": len(X_clean),
    }


def executar_previsao_temporal(df, col_data, col_valor, n_periodos=6, freq="M", agregacao="sum"):
    """
    Executa agregação temporal histórica e regressão linear com projeção para N períodos futuros.

    Parâmetros:
        df: DataFrame com os dados
        col_data: Nome da coluna temporal
        col_valor: Nome da coluna numérica a ser projetada
        n_periodos: Quantidade de períodos futuros a prever (padrão: 6)
        freq: Frequência ('M' para mensal, 'D' para diário, 'Q' para trimestral, 'Y' para anual)
        agregacao: 'sum' ou 'mean'

    Retorna:
        dict com df_historico, df_previsoes, métricas do modelo e resumo executivo.
    """
    if df is None or df.empty:
        raise ValueError("O DataFrame fornecido está vazio.")

    if col_data not in df.columns:
        raise ValueError(f"Coluna temporal '{col_data}' não encontrada no dataset.")

    if col_valor not in df.columns:
        raise ValueError(f"Coluna de valor '{col_valor}' não encontrada no dataset.")

    # Converte coluna de data e valor com tratamento robusto
    df_temp = pd.DataFrame({
        "data_raw": pd.to_datetime(df[col_data], errors="coerce"),
        "valor_raw": pd.to_numeric(df[col_valor], errors="coerce"),
    }).dropna()

    if len(df_temp) < 3:
        raise ValueError(
            "Dados temporais insuficientes após remoção de valores nulos. "
            "São necessárias pelo menos 3 linhas com datas e valores válidos."
        )

    # Cria agrupamento por período
    freq_upper = freq.upper()
    if freq_upper in ("M", "MONTH", "MÊS", "MES"):
        df_temp["periodo"] = df_temp["data_raw"].dt.to_period("M")
        date_offset = pd.DateOffset(months=1)
        date_freq = "MS"
    elif freq_upper in ("D", "DAY", "DIA"):
        df_temp["periodo"] = df_temp["data_raw"].dt.to_period("D")
        date_offset = pd.DateOffset(days=1)
        date_freq = "D"
    elif freq_upper in ("Q", "QUARTER", "TRIMESTRE"):
        df_temp["periodo"] = df_temp["data_raw"].dt.to_period("Q")
        date_offset = pd.DateOffset(months=3)
        date_freq = "QS"
    elif freq_upper in ("Y", "YEAR", "ANO"):
        df_temp["periodo"] = df_temp["data_raw"].dt.to_period("Y")
        date_offset = pd.DateOffset(years=1)
        date_freq = "YS"
    else:
        df_temp["periodo"] = df_temp["data_raw"].dt.to_period("M")
        date_offset = pd.DateOffset(months=1)
        date_freq = "MS"

    # Agregação
    agg_func = "sum" if agregacao == "sum" else "mean"
    df_agrupado = df_temp.groupby("periodo")["valor_raw"].agg(agg_func).reset_index()
    df_agrupado = df_agrupado.sort_values("periodo").reset_index(drop=True)

    if len(df_agrupado) < 3:
        raise ValueError(
            f"O agrupamento temporal gerou apenas {len(df_agrupado)} períodos. "
            "São necessários pelo menos 3 períodos temporais distintos para projeção estatística."
        )

    df_agrupado["data"] = df_agrupado["periodo"].dt.to_timestamp()
    df_agrupado["indice"] = range(len(df_agrupado))
    df_agrupado.rename(columns={"valor_raw": col_valor}, inplace=True)

    # Treina regressão
    X = df_agrupado[["indice"]].values
    y = df_agrupado[col_valor].values
    reg_res = treinar_regressao_linear(X, y)
    df_agrupado["tendencia"] = reg_res["y_pred"]

    # Projeção futura
    n_periodos = max(1, int(n_periodos))
    ultimo_indice = int(df_agrupado["indice"].max())
    indices_futuros = np.arange(ultimo_indice + 1, ultimo_indice + 1 + n_periodos).reshape(-1, 1)
    previsoes_futuras = reg_res["modelo"].predict(indices_futuros)

    ultima_data = df_agrupado["data"].max()
    datas_futuras = pd.date_range(
        start=ultima_data + date_offset,
        periods=n_periodos,
        freq=date_freq
    )

    df_previsoes = pd.DataFrame({
        "data": datas_futuras,
        "valor_previsto": previsoes_futuras,
        "indice": indices_futuros.flatten()
    })
    df_previsoes["crescimento_pct"] = df_previsoes["valor_previsto"].pct_change() * 100

    # Resumo
    total_previsto = float(df_previsoes["valor_previsto"].sum())
    media_mensal = float(df_previsoes["valor_previsto"].mean())
    ultimo_historico = float(df_agrupado[col_valor].iloc[-1])
    crescimento_total = float(
        ((df_previsoes["valor_previsto"].iloc[-1] / ultimo_historico) - 1) * 100
    ) if ultimo_historico != 0 else 0.0

    return {
        "df_historico": df_agrupado,
        "df_previsoes": df_previsoes,
        "modelo": reg_res["modelo"],
        "r2": reg_res["r2"],
        "mae": reg_res["mae"],
        "rmse": reg_res["rmse"],
        "coef": reg_res["coef"],
        "intercept": reg_res["intercept"],
        "total_previsto": total_previsto,
        "media_mensal": media_mensal,
        "ultimo_historico": ultimo_historico,
        "crescimento_total": crescimento_total,
        "col_data": col_data,
        "col_valor": col_valor,
        "n_periodos": n_periodos,
    }


def executar_regressao_bivariada(df, col_x, col_y):
    """
    Executa regressão linear entre duas colunas numéricas quaisquer do dataset (X -> y).
    """
    if df is None or df.empty:
        raise ValueError("O DataFrame fornecido está vazio.")

    if col_x not in df.columns or col_y not in df.columns:
        raise ValueError(f"Colunas '{col_x}' ou '{col_y}' não encontradas no dataset.")

    df_clean = pd.DataFrame({
        col_x: pd.to_numeric(df[col_x], errors="coerce"),
        col_y: pd.to_numeric(df[col_y], errors="coerce"),
    }).dropna()

    if len(df_clean) < 3:
        raise ValueError(
            "Dados insuficientes após remoção de valores nulos para regressão bivariada (mínimo de 3 pares válidos)."
        )

    X = df_clean[[col_x]].values
    y = df_clean[col_y].values
    reg_res = treinar_regressao_linear(X, y)
    df_clean["y_pred"] = reg_res["y_pred"]

    return {
        "df_clean": df_clean,
        "col_x": col_x,
        "col_y": col_y,
        "modelo": reg_res["modelo"],
        "r2": reg_res["r2"],
        "mae": reg_res["mae"],
        "rmse": reg_res["rmse"],
        "coef": reg_res["coef"],
        "intercept": reg_res["intercept"],
        "n_obs": reg_res["n_obs"],
    }
