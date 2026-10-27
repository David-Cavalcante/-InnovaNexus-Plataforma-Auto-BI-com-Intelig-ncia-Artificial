"""
Módulo de Ciência de Dados e Regressão para o InnovaNexus.
Integra recursos de regressão linear (temporal e bivariada), cálculos
estatísticos avançados (quartis, IQR, outliers, coeficiente de variação,
assimetria), intervalos de previsão e projeções futuras, com tratamento
robusto de erros e casos de borda.
"""

import re

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# UTILITÁRIOS DE NOMES DE COLUNAS
# ---------------------------------------------------------
MESES_ABREV_PT = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}


def tokenizar_nome_coluna(nome_coluna):
    """
    Divide o nome de uma coluna em tokens minúsculos (snake_case, camelCase,
    espaços, hífens). Ex.: 'valorPedido_Total' -> ['valor', 'pedido', 'total'].
    Evita falsos positivos de busca por substring (ex.: 'id' dentro de 'cidade').
    """
    if not nome_coluna or not isinstance(nome_coluna, str):
        return []
    texto = re.sub(r"([a-záéíóúâêôãõç])([A-ZÁÉÍÓÚÂÊÔÃÕÇ])", r"\1_\2", nome_coluna.strip())
    return [t for t in re.split(r"[^0-9a-záéíóúâêôãõçà$%]+", texto.lower()) if t]


def eh_coluna_identificador(nome_coluna):
    """
    Indica se a coluna é um identificador/código (id, cod, codigo, uuid, cpf...).
    Usa tokens, portanto 'cidade', 'idade', 'unidade' NÃO são consideradas IDs.
    """
    tokens = tokenizar_nome_coluna(nome_coluna)
    ids = {"id", "ids", "uuid", "guid", "cod", "codigo", "código", "cpf", "cnpj",
           "matricula", "matrícula", "index", "rowid", "pk", "cep", "seq"}
    if any(t in ids for t in tokens):
        return True
    nome = (nome_coluna or "").lower().strip()
    return nome in ("row_id", "id", "index", "codigo", "código") or nome.startswith("id_") or nome.endswith("_id")


def eh_coluna_monetaria(nome_coluna):
    """
    Identifica de forma inteligente se uma coluna representa valores monetários (moeda/R$).
    Evita falsos-positivos em variáveis de quantidade, tempo, idade, scores, etc.
    A análise é feita por tokens (palavras do nome), e não por substring, para que
    nomes como 'receita_liquida' ou 'valor_pedido' não sejam descartados por conter 'id'.
    """
    if not nome_coluna or not isinstance(nome_coluna, str):
        return False

    col = nome_coluna.lower().strip()
    tokens = tokenizar_nome_coluna(nome_coluna)
    if "r$" in col:
        return True

    # Tokens curtos: só bloqueiam se forem a palavra inteira
    nao_monetarios_exatos = {
        "qtd", "qtde", "qt", "num", "n", "nr", "no", "id", "seq", "cep",
        "dia", "dias", "ano", "anos", "mes", "mês", "meses", "pct", "%",
        "age", "day", "days", "year", "years", "month", "months", "rate",
        "count", "load", "span", "grau", "aluno", "periodo", "período",
    }
    # Raízes longas: bloqueiam se o token começar com elas
    nao_monetarios_raizes = (
        "quantidade", "volume", "numero", "número", "hora", "minuto", "segundo",
        "tempo", "idade", "score", "pontuac", "pontuaç", "indice", "índice",
        "ansiedade", "index", "taxa", "percent", "porcent", "foco", "sono",
        "ability", "skill", "intelligence", "speed", "involvement", "quality",
        "wellbeing", "proficiency", "adaptability", "creativity", "thinking",
        "codigo", "código", "hours", "minutes", "unidades", "itens",
    )
    for t in tokens:
        if t in nao_monetarios_exatos or t.startswith(nao_monetarios_raizes):
            return False

    if "nota_fiscal" in col or "total_pago" in col:
        return True

    monetarios_raizes = (
        "faturament", "receita", "despesa", "lucro", "salari", "salár", "ticket",
        "frete", "orcament", "orçament", "dinheiro", "price", "cost", "revenue",
        "fee", "salary", "budget", "sales", "spend", "valor", "preco", "preço",
        "custo", "gasto", "montante", "pagament", "pago", "renda", "amount",
        "margem", "desconto", "comiss", "imposto", "tribut", "multa", "juros",
    )
    return any(t.startswith(monetarios_raizes) for t in tokens)


def obter_prefixo_coluna(nome_coluna):
    """Retorna 'R$ ' se a coluna for monetária, ou '' caso contrário."""
    return "R$ " if eh_coluna_monetaria(nome_coluna) else ""


# ---------------------------------------------------------
# FORMATAÇÃO PT-BR
# ---------------------------------------------------------
def _trocar_separadores(texto):
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_inteiro(valor):
    """Formata inteiros com separador de milhar brasileiro (1.234.567)."""
    if valor is None or pd.isna(valor):
        return "N/A"
    try:
        return f"{int(round(float(valor))):,}".replace(",", ".")
    except (ValueError, TypeError):
        return str(valor)


def formatar_percentual(valor, casas=1, sinal=False):
    """Formata percentuais no padrão brasileiro (12,5% / +3,2%)."""
    if valor is None or pd.isna(valor) or np.isinf(valor):
        return "N/A"
    fmt = f"{{:{'+' if sinal else ''},.{casas}f}}%"
    return _trocar_separadores(fmt.format(float(valor)))


def formatar_decimal(valor, casas=2):
    """Formata número decimal no padrão brasileiro, sem abreviação (1.234,56)."""
    if valor is None or pd.isna(valor):
        return "N/A"
    return _trocar_separadores(f"{float(valor):,.{casas}f}")


def formatar_valor(valor, prefixo=None, nome_coluna=None):
    """
    Formata valores monetários ou numéricos no padrão brasileiro (PT-BR).
    """
    if valor is None or pd.isna(valor):
        return "N/A"
    try:
        val = float(valor)
    except (ValueError, TypeError):
        return str(valor)
    if np.isinf(val):
        return "∞" if val > 0 else "-∞"

    if prefixo is None:
        prefixo = obter_prefixo_coluna(nome_coluna) if nome_coluna else ""

    sinal = "-" if val < 0 else ""
    val_abs = abs(val)

    if val_abs >= 1_000_000:
        texto = f"{prefixo}{sinal}{val_abs / 1_000_000:,.2f}M"
        return _trocar_separadores(texto)
    elif val_abs >= 1_000:
        if val_abs.is_integer() and not prefixo:
            return f"{sinal}{int(val_abs):,}".replace(",", ".")
        texto = f"{prefixo}{sinal}{val_abs:,.2f}" if (val_abs % 1 != 0) else f"{prefixo}{sinal}{val_abs:,.0f}"
        return _trocar_separadores(texto)
    else:
        if val_abs.is_integer() and not prefixo:
            return f"{sinal}{int(val_abs)}"
        texto = f"{prefixo}{sinal}{val_abs:.2f}"
        return texto.replace(".", ",")


def formatar_periodo(data, freq="M"):
    """Formata uma data conforme a periodicidade, com meses em português."""
    if data is None or pd.isna(data):
        return "N/A"
    ts = pd.Timestamp(data)
    f = (freq or "M").upper()
    if f in ("Y", "YS", "A", "ANO", "YEAR"):
        return f"{ts.year}"
    if f in ("Q", "QS", "TRIMESTRE", "QUARTER"):
        return f"T{(ts.month - 1) // 3 + 1}/{ts.year}"
    if f in ("D", "DIA", "DAY"):
        return ts.strftime("%d/%m/%Y")
    if f in ("W", "SEMANA", "WEEK"):
        return f"Sem. {ts.strftime('%d/%m/%Y')}"
    return f"{MESES_ABREV_PT[ts.month]}/{ts.year}"


# ---------------------------------------------------------
# ESTATÍSTICA DESCRITIVA
# ---------------------------------------------------------
def interpretar_assimetria(skew):
    """Interpreta o coeficiente de assimetria (skewness) de Fisher-Pearson."""
    if skew is None or pd.isna(skew):
        return "indeterminada"
    if abs(skew) < 0.5:
        return "aproximadamente simétrica"
    if abs(skew) < 1:
        return "moderadamente assimétrica à direita (cauda de valores altos)" if skew > 0 \
            else "moderadamente assimétrica à esquerda (cauda de valores baixos)"
    return "fortemente assimétrica à direita (cauda de valores altos)" if skew > 0 \
        else "fortemente assimétrica à esquerda (cauda de valores baixos)"


def interpretar_cv(cv):
    """Classifica a dispersão relativa pelo coeficiente de variação."""
    if cv is None or pd.isna(cv):
        return "indeterminada"
    if cv > 50:
        return "alta"
    if cv > 30:
        return "moderada"
    return "baixa"


def calcular_estatisticas_resumo(serie):
    """
    Calcula métricas estatísticas detalhadas de uma série numérica:
    média, desvio padrão, coeficiente de variação, mediana, quartis,
    IQR, assimetria e identificação de outliers (critério IQR) e de
    registros acima do percentil 95.
    """
    s_clean = pd.to_numeric(serie, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s_clean)
    if n == 0:
        raise ValueError("A coluna selecionada não contém valores numéricos válidos.")

    n_total = len(serie)
    media = float(s_clean.mean())
    desvio = float(s_clean.std(ddof=1)) if n > 1 else 0.0
    # CV só é interpretável para variáveis estritamente positivas (escala de razão)
    cv_valido = media != 0 and bool((s_clean >= 0).all())
    coef_var = float((desvio / abs(media) * 100)) if media != 0 else 0.0
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

    mask_inf = s_clean < limite_inf
    mask_sup = s_clean > limite_sup
    outliers_iqr_count = int((mask_inf | mask_sup).sum())
    outliers_iqr_pct = float((outliers_iqr_count / n) * 100) if n > 0 else 0.0

    assimetria = float(s_clean.skew()) if n > 2 and desvio > 0 else 0.0

    return {
        "contagem": n,
        "ausentes": int(n_total - n),
        "media": media,
        "desvio_padrao": desvio,
        "coef_variacao": coef_var,
        "cv_valido": cv_valido,
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
        "outliers_inferiores": int(mask_inf.sum()),
        "outliers_superiores": int(mask_sup.sum()),
        "assimetria": assimetria,
        "assimetria_texto": interpretar_assimetria(assimetria),
    }


def classificar_quartis_outliers(serie):
    """
    Classifica cada elemento da série nas faixas de quartis e outliers (critério IQR).
    Os outliers são separados primeiro, e as faixas de quartil são rotuladas
    pelo intervalo que representam (ex.: 'Q1 (até 25%)').
    """
    s_clean = pd.to_numeric(serie, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
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
        elif v > limite_sup:
            return "Outlier Superior"
        elif v <= q1:
            return "Q1 (até 25%)"
        elif v <= q2:
            return "Q2 (25%–50%)"
        elif v <= q3:
            return "Q3 (50%–75%)"
        else:
            return "Q4 (75%–100%)"

    return s_clean.apply(_classificar)


def interpretar_correlacao(r):
    """Descreve força e direção de um coeficiente de correlação de Pearson."""
    if r is None or pd.isna(r):
        return "indeterminada"
    a = abs(r)
    if a < 0.1:
        forca = "inexistente/desprezível"
    elif a < 0.3:
        forca = "fraca"
    elif a < 0.5:
        forca = "moderada"
    elif a < 0.7:
        forca = "forte"
    else:
        forca = "muito forte"
    if a < 0.1:
        return forca
    return f"{forca} {'positiva' if r > 0 else 'negativa'}"


def calcular_correlacao(serie_x, serie_y):
    """
    Correlação de Pearson com p-valor e tamanho amostral, ignorando pares inválidos.
    Retorna dict com r, p_valor, n e interpretação.
    """
    df_par = pd.DataFrame({
        "x": pd.to_numeric(serie_x, errors="coerce"),
        "y": pd.to_numeric(serie_y, errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df_par)
    if n < 3 or df_par["x"].nunique() < 2 or df_par["y"].nunique() < 2:
        return {"r": np.nan, "p_valor": np.nan, "n": n, "interpretacao": "indeterminada", "significativa": False}
    r, p = sp_stats.pearsonr(df_par["x"], df_par["y"])
    return {
        "r": float(r),
        "p_valor": float(p),
        "n": n,
        "interpretacao": interpretar_correlacao(r),
        "significativa": bool(p < 0.05),
    }


def ranking_correlacoes(df, colunas, top=5, min_obs=10):
    """
    Retorna os pares de variáveis numéricas com maiores correlações absolutas,
    com p-valor. Ignora pares com poucas observações.
    """
    pares = []
    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):
            res = calcular_correlacao(df[colunas[i]], df[colunas[j]])
            if res["n"] >= min_obs and not pd.isna(res["r"]):
                pares.append({"var_1": colunas[i], "var_2": colunas[j], **res})
    pares.sort(key=lambda p: abs(p["r"]), reverse=True)
    return pares[:top]


def resumir_por_categoria(df, col_cat, col_num, agregacao="mean", min_registros=1):
    """
    Agrega uma métrica por categoria, sempre informando a contagem de registros
    de cada grupo (base da média) e a participação percentual no total.
    agregacao: 'mean' | 'sum' | 'median' | 'count'
    """
    base = pd.DataFrame({
        col_cat: df[col_cat].astype("object").where(df[col_cat].notna(), "Não informado").astype(str),
        col_num: pd.to_numeric(df[col_num], errors="coerce"),
    })
    grp = base.groupby(col_cat)[col_num]
    res = pd.DataFrame({
        "registros": base.groupby(col_cat).size(),
        "valores_validos": grp.count(),
        "soma": grp.sum(min_count=1),
        "media": grp.mean(),
        "mediana": grp.median(),
    }).reset_index()

    mapa = {"mean": "media", "sum": "soma", "median": "mediana", "count": "registros"}
    col_valor = mapa.get(agregacao, "media")
    res["valor"] = res[col_valor]
    res = res[res["registros"] >= max(1, int(min_registros))]

    total_soma = res["soma"].sum()
    total_reg = res["registros"].sum()
    res["participacao_soma_pct"] = (res["soma"] / total_soma * 100) if total_soma else np.nan
    res["participacao_registros_pct"] = (res["registros"] / total_reg * 100) if total_reg else np.nan
    return res.sort_values("valor", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------
# REGRESSÃO
# ---------------------------------------------------------
def treinar_regressao_linear(X, y):
    """
    Ajusta modelo LinearRegression sobre X e y com validação de dados ausentes,
    tamanho mínimo amostral e variância zero.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    if len(X_arr) != len(y_arr):
        raise ValueError(
            f"Dimensões incompatíveis: X tem {len(X_arr)} itens e y tem {len(y_arr)} itens."
        )

    mask_valida = ~(
        np.isnan(X_arr).any(axis=1) | np.isinf(X_arr).any(axis=1) |
        np.isnan(y_arr) | np.isinf(y_arr)
    )

    X_clean = X_arr[mask_valida]
    y_clean = y_arr[mask_valida]

    if len(X_clean) < 3:
        raise ValueError(
            "Dados insuficientes para regressão linear. São necessárias pelo menos 3 observações válidas."
        )

    if np.all(X_clean == X_clean[0]):
        raise ValueError(
            "A variável preditora (X) possui variância zero (todos os valores são idênticos). "
            "Não é possível traçar uma reta de regressão."
        )

    modelo = LinearRegression()
    modelo.fit(X_clean, y_clean)
    y_pred = modelo.predict(X_clean)

    r2 = float(r2_score(y_clean, y_pred)) if np.var(y_clean) > 0 else 1.0
    mae = float(mean_absolute_error(y_clean, y_pred))
    mse = float(mean_squared_error(y_clean, y_pred))
    rmse = float(np.sqrt(mse))

    coef = float(modelo.coef_[0]) if hasattr(modelo.coef_, "__iter__") else float(modelo.coef_)
    intercept = float(modelo.intercept_)

    # Inferência estatística (regressão simples): erro padrão, p-valor da inclinação
    n = len(X_clean)
    x1 = X_clean[:, 0]
    sxx = float(np.sum((x1 - x1.mean()) ** 2))
    gl = n - 2
    sse = float(np.sum((y_clean - y_pred) ** 2))
    erro_padrao_residual = float(np.sqrt(sse / gl)) if gl > 0 else np.nan
    if gl > 0 and sxx > 0 and erro_padrao_residual > 0:
        se_coef = erro_padrao_residual / np.sqrt(sxx)
        t_stat = coef / se_coef
        p_valor = float(2 * sp_stats.t.sf(abs(t_stat), gl))
    else:
        p_valor = 0.0 if erro_padrao_residual == 0 else np.nan

    return {
        "modelo": modelo,
        "y_pred": y_pred,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "coef": coef,
        "intercept": intercept,
        "p_valor_coef": p_valor,
        "erro_padrao_residual": erro_padrao_residual,
        "x_media": float(x1.mean()),
        "sxx": sxx,
        "X_clean": X_clean,
        "y_clean": y_clean,
        "n_obs": n,
    }


def calcular_intervalo_previsao(reg_res, x_novos, confianca=0.95):
    """
    Calcula o intervalo de previsão (para uma nova observação) de uma
    regressão linear simples: ŷ ± t * s * sqrt(1 + 1/n + (x0 - x̄)² / Sxx).
    """
    x0 = np.asarray(x_novos, dtype=float).reshape(-1)
    y_hat = reg_res["modelo"].predict(x0.reshape(-1, 1))
    n = reg_res["n_obs"]
    s = reg_res["erro_padrao_residual"]
    if n <= 2 or pd.isna(s) or reg_res["sxx"] <= 0:
        return y_hat, y_hat.copy(), y_hat.copy()
    t_crit = sp_stats.t.ppf(1 - (1 - confianca) / 2, n - 2)
    margem = t_crit * s * np.sqrt(1 + 1 / n + (x0 - reg_res["x_media"]) ** 2 / reg_res["sxx"])
    return y_hat, y_hat - margem, y_hat + margem


def converter_serie_temporal_segura(serie_tempo):
    """
    Converte uma série em datetime de forma inteligente.
    Se detectar números inteiros no intervalo de anos (1800 a 2200),
    converte para YYYY-01-01 em vez de nanossegundos Unix (1970).
    Também aceita datas no padrão brasileiro (dd/mm/aaaa).
    """
    if pd.api.types.is_datetime64_any_dtype(serie_tempo):
        return serie_tempo

    s_num = pd.to_numeric(serie_tempo, errors="coerce")
    s_num_validos = s_num.dropna()

    if len(s_num_validos) > 0 and s_num_validos.between(1800, 2200).all() and (s_num_validos % 1 == 0).all():
        anos_str = s_num.astype("Int64").astype("string")
        return pd.to_datetime(anos_str + "-01-01", errors="coerce")

    return converter_texto_para_data(serie_tempo)


def converter_texto_para_data(serie):
    """
    Converte texto em datas testando formatos ISO e brasileiro (dia primeiro),
    escolhendo o que reconhece mais valores. Evita a inversão dia/mês.
    """
    if pd.api.types.is_datetime64_any_dtype(serie):
        return serie
    s = serie.astype("string").str.strip()
    melhor = None
    melhor_validos = -1
    for kwargs in ({"format": "ISO8601"}, {"dayfirst": True, "format": "mixed"}, {"format": "mixed"}):
        try:
            conv = pd.to_datetime(s, errors="coerce", **kwargs)
        except (ValueError, TypeError):
            continue
        validos = int(conv.notna().sum())
        if validos > melhor_validos:
            melhor, melhor_validos = conv, validos
    return melhor if melhor is not None else pd.to_datetime(s, errors="coerce")


_CONFIG_FREQ = {
    "M": ("M", "MS", "M"),
    "D": ("D", "D", "D"),
    "W": ("W", "W-MON", "W"),
    "Q": ("Q", "QS", "Q"),
    "Y": ("Y", "YS", "Y"),
}


def _normalizar_freq(freq):
    f = (freq or "M").upper()
    if f in ("D", "DAY", "DIA"):
        return "D"
    if f in ("W", "SEMANA", "WEEK"):
        return "W"
    if f in ("Q", "QS", "QUARTER", "TRIMESTRE"):
        return "Q"
    if f in ("Y", "YS", "YEAR", "ANO", "A"):
        return "Y"
    return "M"


def executar_previsao_temporal(df, col_data, col_valor, n_periodos=6, freq="M", agregacao="sum",
                               remover_periodo_incompleto=True, confianca=0.95):
    """
    Executa agregação temporal histórica e regressão linear com projeção para N períodos futuros.

    Melhorias de correção estatística:
    - O eixo X é a posição real do período no calendário (lacunas sem dados
      não "encolhem" o tempo).
    - Detecta e (opcionalmente) remove o último período parcial, que distorce
      a tendência quando a agregação é soma.
    - Retorna intervalo de previsão de 95% para cada período projetado.
    """
    if df is None or df.empty:
        raise ValueError("O DataFrame fornecido está vazio.")

    if col_data not in df.columns:
        raise ValueError(f"Coluna temporal '{col_data}' não encontrada no dataset.")

    if col_valor not in df.columns:
        raise ValueError(f"Coluna de valor '{col_valor}' não encontrada no dataset.")

    datas_convertidas = converter_serie_temporal_segura(df[col_data])

    df_temp = pd.DataFrame({
        "data_raw": datas_convertidas,
        "valor_raw": pd.to_numeric(df[col_valor], errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_temp) < 3:
        raise ValueError(
            "Dados temporais insuficientes após remoção de valores nulos. "
            "São necessárias pelo menos 3 linhas com datas e valores válidos."
        )

    f = _normalizar_freq(freq)
    period_code, date_freq, fmt_code = _CONFIG_FREQ[f]
    df_temp["periodo"] = df_temp["data_raw"].dt.to_period(period_code)

    agg_func = {"sum": "sum", "mean": "mean", "median": "median", "count": "count"}.get(agregacao, "mean")
    df_agrupado = (
        df_temp.groupby("periodo")
        .agg(valor_raw=("valor_raw", agg_func), registros=("valor_raw", "size"), data_max=("data_raw", "max"))
        .reset_index()
        .sort_values("periodo")
        .reset_index(drop=True)
    )

    # Detecção de período final incompleto (ex.: mês corrente com poucos dias)
    periodo_incompleto_removido = None
    if len(df_agrupado) >= 4 and f != "D":
        inicio = df_agrupado["periodo"].dt.start_time
        fim = df_agrupado["periodo"].dt.end_time
        duracao = (fim - inicio).dt.total_seconds()
        cobertura = ((df_agrupado["data_max"] - inicio).dt.total_seconds() / duracao).clip(0, 1)
        cobertura_tipica = float(cobertura.iloc[:-1].median())
        # Só faz sentido quando os dados se espalham dentro dos períodos (não são datas "dia 1")
        if cobertura_tipica > 0.5 and float(cobertura.iloc[-1]) < 0.5 * cobertura_tipica:
            ultimo = df_agrupado.iloc[-1]
            periodo_incompleto_removido = {
                "periodo": ultimo["periodo"].start_time,
                "valor": float(ultimo["valor_raw"]),
                "cobertura_pct": float(cobertura.iloc[-1] * 100),
            }
            if remover_periodo_incompleto and agregacao in ("sum", "count"):
                df_agrupado = df_agrupado.iloc[:-1].reset_index(drop=True)
            else:
                periodo_incompleto_removido["mantido"] = True

    if len(df_agrupado) < 3:
        raise ValueError(
            f"O agrupamento temporal gerou apenas {len(df_agrupado)} período(s). "
            "Dados insuficientes: são necessários pelo menos 3 períodos distintos para projeção estatística. "
            "Tente uma periodicidade menor (ex.: Mensal em vez de Anual)."
        )

    primeiro_periodo = df_agrupado["periodo"].iloc[0]
    df_agrupado["indice"] = [(p - primeiro_periodo).n for p in df_agrupado["periodo"]]
    total_periodos_calendario = int(df_agrupado["indice"].max()) + 1
    periodos_sem_dados = total_periodos_calendario - len(df_agrupado)

    df_agrupado["data"] = df_agrupado["periodo"].dt.to_timestamp()
    df_agrupado.rename(columns={"valor_raw": col_valor}, inplace=True)
    df_agrupado.drop(columns=["data_max"], inplace=True)

    X = df_agrupado[["indice"]].values
    y = df_agrupado[col_valor].values
    reg_res = treinar_regressao_linear(X, y)
    df_agrupado["tendencia"] = reg_res["y_pred"]

    n_periodos = max(1, int(n_periodos))
    ultimo_periodo = df_agrupado["periodo"].iloc[-1]
    ultimo_indice = int(df_agrupado["indice"].iloc[-1])
    indices_futuros = np.arange(ultimo_indice + 1, ultimo_indice + 1 + n_periodos)
    previsoes, lim_inf, lim_sup = calcular_intervalo_previsao(reg_res, indices_futuros, confianca)
    datas_futuras = [(ultimo_periodo + k).start_time for k in range(1, n_periodos + 1)]

    df_previsoes = pd.DataFrame({
        "data": pd.to_datetime(datas_futuras),
        "valor_previsto": previsoes,
        "limite_inferior": lim_inf,
        "limite_superior": lim_sup,
        "indice": indices_futuros,
    })

    ultimo_historico = float(df_agrupado[col_valor].iloc[-1])
    ultima_tendencia = float(df_agrupado["tendencia"].iloc[-1])
    df_previsoes["crescimento_pct"] = df_previsoes["valor_previsto"].pct_change() * 100
    if ultimo_historico != 0:
        df_previsoes.loc[0, "crescimento_pct"] = (
            (df_previsoes.loc[0, "valor_previsto"] - ultimo_historico) / abs(ultimo_historico)
        ) * 100

    total_previsto = float(df_previsoes["valor_previsto"].sum())
    media_mensal = float(df_previsoes["valor_previsto"].mean())
    crescimento_total = float(
        ((df_previsoes["valor_previsto"].iloc[-1] - ultimo_historico) / abs(ultimo_historico)) * 100
    ) if ultimo_historico != 0 else 0.0
    # Variação pela própria reta (menos sensível a um último ponto atípico)
    crescimento_tendencia = float(
        ((df_previsoes["valor_previsto"].iloc[-1] - ultima_tendencia) / abs(ultima_tendencia)) * 100
    ) if ultima_tendencia != 0 else 0.0

    valores_negativos_previstos = bool((df_previsoes["valor_previsto"] < 0).any()) and bool((y >= 0).all())

    return {
        "df_historico": df_agrupado,
        "df_previsoes": df_previsoes,
        "modelo": reg_res["modelo"],
        "r2": reg_res["r2"],
        "mae": reg_res["mae"],
        "rmse": reg_res["rmse"],
        "coef": reg_res["coef"],
        "intercept": reg_res["intercept"],
        "p_valor_coef": reg_res["p_valor_coef"],
        "tendencia_significativa": bool(reg_res["p_valor_coef"] < 0.05) if not pd.isna(reg_res["p_valor_coef"]) else False,
        "total_previsto": total_previsto,
        "media_mensal": media_mensal,
        "ultimo_historico": ultimo_historico,
        "crescimento_total": crescimento_total,
        "crescimento_tendencia": crescimento_tendencia,
        "col_data": col_data,
        "col_valor": col_valor,
        "n_periodos": n_periodos,
        "n_periodos_historicos": len(df_agrupado),
        "periodos_sem_dados": int(periodos_sem_dados),
        "periodo_incompleto": periodo_incompleto_removido,
        "valores_negativos_previstos": valores_negativos_previstos,
        "confianca": confianca,
        "freq": fmt_code,
        "agregacao": agregacao,
        "formato_data": {"Y": "%Y", "D": "%d/%m/%Y"}.get(f, "%b/%Y"),
    }


def executar_regressao_bivariada(df, col_x, col_y):
    """
    Executa regressão linear entre duas colunas numéricas quaisquer do dataset (X -> y).
    """
    if df is None or df.empty:
        raise ValueError("O DataFrame fornecido está vazio.")

    if col_x not in df.columns or col_y not in df.columns:
        raise ValueError(f"Colunas '{col_x}' ou '{col_y}' não encontradas no dataset.")

    if col_x == col_y:
        raise ValueError("Selecione colunas diferentes para X e Y.")

    df_clean = pd.DataFrame({
        col_x: pd.to_numeric(df[col_x], errors="coerce"),
        col_y: pd.to_numeric(df[col_y], errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_clean) < 3:
        raise ValueError(
            "Dados insuficientes após remoção de valores nulos para regressão bivariada (mínimo de 3 pares válidos)."
        )

    X = df_clean[[col_x]].values
    y = df_clean[col_y].values
    reg_res = treinar_regressao_linear(X, y)
    df_clean["y_pred"] = reg_res["y_pred"]
    corr = calcular_correlacao(df_clean[col_x], df_clean[col_y])

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
        "p_valor_coef": reg_res["p_valor_coef"],
        "r_pearson": corr["r"],
        "interpretacao": corr["interpretacao"],
        "significativa": bool(reg_res["p_valor_coef"] < 0.05) if not pd.isna(reg_res["p_valor_coef"]) else False,
        "n_obs": reg_res["n_obs"],
    }
