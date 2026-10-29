import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import tempfile
from io import StringIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from openai import OpenAI
from fpdf import FPDF
import numpy as np

import csv
import hashlib
import re

from data_science import (
    formatar_valor,
    formatar_inteiro,
    formatar_percentual,
    formatar_decimal,
    formatar_r2,
    interpretar_r2,
    formatar_periodo,
    eh_coluna_monetaria,
    eh_coluna_identificador,
    tokenizar_nome_coluna,
    obter_prefixo_coluna,
    calcular_estatisticas_resumo,
    classificar_quartis_outliers,
    calcular_correlacao,
    ranking_correlacoes,
    resumir_por_categoria,
    interpretar_cv,
    converter_texto_para_data,
    executar_previsao_temporal,
    executar_regressao_bivariada
)

# ---------------------------------------------------------
# CONFIGURAÇÃO DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="InnovaNexus - Insights & Dashboards com IA",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)

CHAVES_PADRAO = {
    "Groq (Rápido/Gratuito)": "",
    "Gemini (Google)":        "",
}

MESES_PT = {
    'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março',
    'April': 'Abril', 'May': 'Maio', 'June': 'Junho',
    'July': 'Julho', 'August': 'Agosto', 'September': 'Setembro',
    'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
}

# ---------------------------------------------------------
# GERENCIAMENTO DE SESSÃO & AUTENTICAÇÃO
# ---------------------------------------------------------
if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False

def realizar_login(usuario, senha):
    if usuario == "admin" and senha == "1234":
        st.session_state['autenticado'] = True
        st.success("Login efetuado com sucesso!")
        st.rerun()
    else:
        st.error("Utilizador ou palavra-passe incorretos.")

def realizar_logout():
    st.session_state['autenticado'] = False
    st.rerun()

# ---------------------------------------------------------
# FUNÇÃO DE GERAÇÃO DE PDF EXECUTIVO
# ---------------------------------------------------------
class RelatorioPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, 'InnovaNexus - Relatorio Executivo de Dados', border=False, new_x="LMARGIN", new_y="NEXT", align='L')
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, 'Parecer Prescritivo de Negocio, Diagnostico de ETL e Visualizacao de Indicadores', border=False, new_x="LMARGIN", new_y="NEXT", align='L')
        self.ln(5)
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Pagina {self.page_no()} | Plataforma InnovaNexus Auto-BI', align='C')

def limpar_texto_pdf(texto):
    texto = re.sub(r"<[^>]+>", "", texto)
    substituicoes = {
        '•': '-', '–': '-', '—': '-', '“': '"', '”': '"', '‘': "'", '’': "'",
        '×': 'x', '÷': '/', '≤': '<=', '≥': '>=', '≈': '~', '≠': '!=', '±': '+/-',
        '²': '2', '³': '3', 'º': 'o', 'ª': 'a', 'η': 'eta', '∞': 'inf',
        '🚀': '', '🎯': '', '📌': '', '💡': '', '✨': '', '🧠': '', '🔬': '',
        '👀': '', '📊': '', '⚠️': 'ALERTA:', '📈': '', '📉': '', '🔑': '',
        '✅': '', '❌': '', '📋': '', '🏆': '', '💰': '', '📅': '', 'ℹ️': 'INFO:'
    }
    for orig, subst in substituicoes.items():
        texto = texto.replace(orig, subst)
    return texto.encode('latin-1', 'replace').decode('latin-1')

def gerar_pdf_executivo(total_linhas, total_colunas, nulos_tratados, parecer_texto, df, var_num, var_cat):
    pdf = RelatorioPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, '1. RESUMO DE ENGENHARIA E ESTRUTURA DOS DADOS', new_x="LMARGIN", new_y="NEXT")

    pdf.set_font('Helvetica', '', 10)
    pdf.cell(65, 7, f'- Total de Registros: {formatar_inteiro(total_linhas)}', border=0)
    pdf.cell(55, 7, f'- Total de Colunas: {total_colunas}', border=0)
    pdf.cell(70, 7, f'- Ausentes na Origem: {formatar_inteiro(nulos_tratados)}', border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, '2. PAINEL VISUAL DE INDICADORES CHAVE', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    tmp_path = None
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        res_top = resumir_por_categoria(df, var_cat, var_num, 'mean').head(8)
        ax1.bar(res_top[var_cat].astype(str), res_top['media'], color='#2563eb')
        ax1.set_title(limpar_texto_pdf(f'Media de {var_num} por {var_cat}'), fontsize=8, fontweight='bold')
        ax1.tick_params(axis='x', rotation=35, labelsize=6)
        ax1.tick_params(axis='y', labelsize=6)

        # Pizza usa a soma real (ou contagem se a soma não for positiva), nunca média
        vals_pizza = res_top['soma'] if (res_top['soma'] > 0).any() else res_top['registros']
        titulo_pizza = f'Participacao no Total de {var_num}' if (res_top['soma'] > 0).any() else f'Participacao nos Registros ({var_cat})'
        vals_pos = vals_pizza.clip(lower=0)
        if vals_pos.sum() > 0:
            ax2.pie(vals_pos, labels=res_top[var_cat].astype(str), autopct='%1.1f%%', textprops={'fontsize': 6})
        ax2.set_title(limpar_texto_pdf(titulo_pizza), fontsize=8, fontweight='bold')

        plt.tight_layout()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            tmp_path = tmpfile.name
        plt.savefig(tmp_path, dpi=150)
        plt.close(fig)
        pdf.image(tmp_path, x=10, y=pdf.get_y(), w=190)
        pdf.ln(68)
    except Exception as e:
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 8, limpar_texto_pdf(f'(Graficos omitidos: {e})'), new_x="LMARGIN", new_y="NEXT")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, '3. PARECER EXECUTIVO E RECOMENDACOES (INNOVANEXUS AI)', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font('Helvetica', '', 9.5)
    pdf.set_text_color(51, 65, 85)
    texto_processado = parecer_texto.replace('#', '').replace('*', '').replace('`', '')
    pdf.multi_cell(0, 5, limpar_texto_pdf(texto_processado))

    saida = pdf.output()
    return bytes(saida) if isinstance(saida, (bytearray, bytes)) else saida.encode('latin-1')

# ---------------------------------------------------------
# UTILITÁRIOS DE TIPOS DE COLUNAS
# ---------------------------------------------------------
def colunas_texto(df):
    """Colunas textuais (compatível com pandas 2 'object' e pandas 3 'str')."""
    return [
        c for c in df.columns
        if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]


def colunas_categoricas(df):
    """Colunas textuais/categóricas, excluindo identificadores."""
    cats = colunas_texto(df) + [
        c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    return [c for c in cats if not eh_coluna_identificador(c)]


def colunas_numericas_analiticas(df):
    """Colunas numéricas úteis para análise (exclui IDs, códigos e colunas derivadas de ano)."""
    todas = [c for c in df.select_dtypes(include=['number']).columns if not pd.api.types.is_bool_dtype(df[c])]
    uteis = [
        c for c in todas
        if not eh_coluna_identificador(c) and not c.lower().endswith('_ano')
        and df[c].nunique(dropna=True) > 1
    ]
    return uteis or todas


_PADRAO_NUMERO_BR = re.compile(r"^-?(R\$\s?)?-?\d{1,3}(\.\d{3})*(,\d+)?%?$|^-?(R\$\s?)?-?\d+(,\d+)?%?$")


def _converter_texto_numerico(serie):
    """
    Converte textos numéricos em número, reconhecendo formatos brasileiros
    ('R$ 1.234,56', '12,5%', '1.000') e internacionais ('1234.56').
    Retorna None quando a coluna não é majoritariamente numérica.
    """
    s = serie.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return None

    # Formato internacional simples
    conv_int = pd.to_numeric(s.str.replace(r"^R\$\s?", "", regex=True).str.rstrip('%'), errors='coerce')
    taxa_int = conv_int.notna().mean()

    # Formato brasileiro (ponto = milhar, vírgula = decimal)
    eh_br = s.str.match(_PADRAO_NUMERO_BR)
    taxa_br = eh_br.mean() if s.str.contains(',', regex=False).any() else 0.0

    def _limpar_br(x):
        return (x.astype(str).str.strip()
                .str.replace(r"^R\$\s?", "", regex=True).str.replace("R$", "", regex=False)
                .str.rstrip('%').str.replace('.', '', regex=False).str.replace(',', '.', regex=False))

    if taxa_br >= 0.9 and taxa_br >= taxa_int:
        return pd.to_numeric(_limpar_br(serie.where(serie.notna(), None)), errors='coerce')
    if taxa_int >= 0.9:
        limpo = serie.astype(str).str.strip().str.replace(r"^R\$\s?", "", regex=True).str.rstrip('%')
        return pd.to_numeric(limpo.where(serie.notna(), None), errors='coerce')
    return None


def ler_csv_robusto(arquivo):
    """
    Lê CSV detectando automaticamente codificação (UTF-8 / Latin-1) e
    separador (',', ';', tab, '|'), comum em arquivos exportados do Excel PT-BR.
    """
    bruto = arquivo.getvalue() if hasattr(arquivo, 'getvalue') else arquivo.read()
    texto = None
    for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1'):
        try:
            texto = bruto.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if texto is None:
        texto = bruto.decode('utf-8', errors='replace')

    # Corrige linhas inteiras envolvidas em aspas (exportações quebradas)
    linhas = texto.splitlines()
    if linhas and all(l.startswith('"') and l.endswith('"') and l.count('"') == 2 for l in linhas[:20] if l):
        linhas = [l[1:-1] if l.startswith('"') and l.endswith('"') else l for l in linhas]
        texto = "\n".join(linhas)

    amostra = "\n".join(texto.splitlines()[:50])
    sep = ','
    try:
        sep = csv.Sniffer().sniff(amostra, delimiters=[',', ';', '\t', '|']).delimiter
    except csv.Error:
        contagens = {d: amostra.count(d) for d in [';', ',', '\t', '|']}
        sep = max(contagens, key=contagens.get) if any(contagens.values()) else ','

    return pd.read_csv(StringIO(texto), sep=sep)


def _parece_coluna_data(nome_coluna):
    tokens = tokenizar_nome_coluna(nome_coluna)
    chaves = {'date', 'data', 'dt', 'datetime', 'timestamp', 'dia', 'periodo', 'período', 'competencia',
              'competência', 'created', 'updated', 'vencimento', 'emissao', 'emissão'}
    duracoes = {'horas', 'minutos', 'segundos', 'hours', 'minutes', 'seconds', 'idade', 'age', 'tempo', 'dias'}
    if any(t in duracoes for t in tokens):
        return False
    return any(t in chaves or t.startswith(('data', 'date')) for t in tokens) or \
        nome_coluna.lower().endswith(('_at', '_em', '_date', '_dt'))


# ---------------------------------------------------------
# FUNÇÃO DE ETL AUTOMÁTICO (COM AUDITORIA DE NULOS REAIS)
# ---------------------------------------------------------
def executar_etl_automatico(df_raw, imputar_numericos=False, remover_duplicatas=True):
    """
    ETL automático:
    - Remove duplicatas exatas (opcional) e espaços extras.
    - Converte textos numéricos (inclusive formato brasileiro 'R$ 1.234,56').
    - Converte colunas de data somente quando >= 80% dos valores são datas válidas.
    - Valores ausentes numéricos são MANTIDOS por padrão (não distorcem média,
      desvio padrão, outliers e correlações). Imputação pela mediana é opcional.
    - Datas ausentes NUNCA são preenchidas (evita criar datas fictícias).
    """
    df_clean = df_raw.copy()
    df_clean.columns = [str(c).strip() for c in df_clean.columns]
    duplicatas_encontradas = int(df_clean.duplicated().sum())
    if remover_duplicatas:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    for col in colunas_texto(df_clean):
        serie = df_clean[col].astype('object')
        df_clean[col] = serie.where(serie.isna(), serie.astype(str).str.strip())
        df_clean[col] = df_clean[col].replace({'': np.nan, 'nan': np.nan, 'NaN': np.nan, 'None': np.nan,
                                               'null': np.nan, 'NULL': np.nan, 'N/A': np.nan, '-': np.nan})

    nulos_originais = df_clean.isnull().sum().to_dict()
    conversoes = []

    # Datas: só converte se a maior parte dos valores for reconhecida
    for col in colunas_texto(df_clean):
        if _parece_coluna_data(col) and df_clean[col].notna().any():
            try:
                conv = converter_texto_para_data(df_clean[col])
                taxa = conv.notna().sum() / max(1, df_clean[col].notna().sum())
                if taxa >= 0.8:
                    df_clean[col] = conv
                    conversoes.append(f"'{col}' → data ({taxa * 100:.0f}% reconhecidos)")
            except Exception:
                pass

    # Números armazenados como texto (inclui formato brasileiro)
    for col in colunas_texto(df_clean):
        if eh_coluna_identificador(col):
            continue
        conv = _converter_texto_numerico(df_clean[col])
        if conv is not None:
            df_clean[col] = conv
            conversoes.append(f"'{col}' → número")

    if 'data_hora_real' in df_clean.columns and 'data_hora_agendada' in df_clean.columns:
        if (pd.api.types.is_datetime64_any_dtype(df_clean['data_hora_real']) and
                pd.api.types.is_datetime64_any_dtype(df_clean['data_hora_agendada'])):
            df_clean['atraso_real_minutos'] = (
                (df_clean['data_hora_real'] - df_clean['data_hora_agendada'])
                .dt.total_seconds() / 60.0
            ).round(1)

    if 'flag_pontualidade' in df_clean.columns:
        # Valores desconhecidos permanecem ausentes (não são contados como atraso)
        df_clean['pontualidade_pct'] = (
            df_clean['flag_pontualidade'].astype(str).str.strip().str.lower()
            .map({'true': 100, 'false': 0, '1': 100, '0': 0, 'sim': 100, 'não': 0, 'nao': 0,
                  '1.0': 100, '0.0': 0})
        )

    imputados = {}
    for col in df_clean.columns:
        n_nulos = int(df_clean[col].isnull().sum())
        if n_nulos > 0:
            if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                continue  # nunca inventar datas
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                if imputar_numericos:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    imputados[col] = n_nulos
            else:
                df_clean[col] = df_clean[col].fillna("Não informado")
                imputados[col] = n_nulos

    colunas_datas = [c for c in df_clean.columns if pd.api.types.is_datetime64_any_dtype(df_clean[c])]
    for col in colunas_datas:
        prefixo = col.replace('_datetime', '').replace('_date', '').replace('_data', '')
        df_clean[f'{prefixo}_ano'] = df_clean[col].dt.year.astype('Int64')
        # Formato "01 - Janeiro" mantém a ordem cronológica nos gráficos
        df_clean[f'{prefixo}_mes'] = df_clean[col].apply(
            lambda d: f"{d.month:02d} - {list(MESES_PT.values())[d.month - 1]}" if pd.notna(d) else "Não informado"
        )
        nulos_originais.setdefault(f'{prefixo}_ano', 0)
        nulos_originais.setdefault(f'{prefixo}_mes', 0)

    df_clean.attrs['nulos_originais'] = nulos_originais
    df_clean.attrs['duplicatas_removidas'] = duplicatas_encontradas if remover_duplicatas else 0
    df_clean.attrs['duplicatas_encontradas'] = duplicatas_encontradas
    df_clean.attrs['imputados'] = imputados
    df_clean.attrs['conversoes'] = conversoes
    df_clean.attrs['imputar_numericos'] = imputar_numericos
    return df_clean

# ---------------------------------------------------------
# FUNÇÃO DE CARREGAMENTO SQL
# ---------------------------------------------------------
def carregar_dados_sql(db_type, host, port, user, password, database, tabela):
    try:
        from sqlalchemy.engine import URL
        # Apenas nomes de tabela válidos (evita injeção de SQL): schema.tabela
        if not re.fullmatch(r"[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)?", (tabela or "").strip()):
            return None, "Nome de tabela inválido. Use apenas letras, números e '_' (opcionalmente 'schema.tabela')."
        tabela = tabela.strip()
        porta = int(port) if str(port).strip().isdigit() else None
        if db_type == "PostgreSQL":
            # URL.create trata corretamente senhas com caracteres especiais (@, :, /)
            url = URL.create("postgresql+psycopg2", username=user, password=password, host=host, port=porta, database=database)
        elif db_type == "MySQL":
            url = URL.create("mysql+pymysql", username=user, password=password, host=host, port=porta, database=database)
        elif db_type == "SQLite":
            url = f"sqlite:///{database}"
        else:
            return None, f"Tipo de banco não suportado: {db_type}"

        engine = create_engine(url)
        query = f"SELECT * FROM {tabela}"
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df, None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# MOTOR ANALÍTICO NATIVO (CONTINGÊNCIA / OFFLINE BASEADO NOS DADOS)
# ---------------------------------------------------------
def _escolher_metrica_principal(df, col_num):
    """Prioriza métricas monetárias; senão, a primeira métrica com variação."""
    monetarias = [c for c in col_num if eh_coluna_monetaria(c)]
    return (monetarias or col_num or [None])[0]


def _escolher_categoria_principal(df, col_cat, metrica):
    """
    Escolhe a categoria que mais explica a variação da métrica (eta², razão de
    correlação), considerando apenas categorias com 2 a 30 grupos.
    """
    candidatas = [c for c in col_cat if 2 <= df[c].nunique(dropna=True) <= 30]
    if not candidatas or metrica is None:
        return (candidatas or col_cat or [None])[0], None
    y = pd.to_numeric(df[metrica], errors='coerce')
    var_total = ((y - y.mean()) ** 2).sum()
    melhor, melhor_eta = candidatas[0], -1.0
    for c in candidatas:
        tmp = pd.DataFrame({'g': df[c].astype(str), 'y': y}).dropna()
        if tmp.empty or var_total == 0:
            continue
        medias = tmp.groupby('g')['y'].agg(['mean', 'count'])
        var_entre = (medias['count'] * (medias['mean'] - tmp['y'].mean()) ** 2).sum()
        eta2 = float(var_entre / ((tmp['y'] - tmp['y'].mean()) ** 2).sum()) if ((tmp['y'] - tmp['y'].mean()) ** 2).sum() else 0.0
        if eta2 > melhor_eta:
            melhor, melhor_eta = c, eta2
    return melhor, (melhor_eta if melhor_eta >= 0 else None)


def construir_perfil_dados(df, col_num=None, col_cat=None, col_data=None, max_cats=8):
    """
    Gera um perfil textual compacto e FIEL do dataset (tipos, ausentes,
    estatísticas, correlações com p-valor, resumo por categoria e período),
    usado tanto pelo prompt das IAs externas quanto pela auditoria.
    """
    col_num = col_num if col_num is not None else colunas_numericas_analiticas(df)
    col_cat = col_cat if col_cat is not None else colunas_categoricas(df)
    col_data = col_data if col_data is not None else [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    nulos_orig = df.attrs.get('nulos_originais', {})
    partes = [f"Registros: {len(df)} | Colunas: {df.shape[1]} | Duplicatas removidas: {df.attrs.get('duplicatas_removidas', 0)}"]

    partes.append("\n[COLUNAS] nome | tipo | ausentes_originais | valores_unicos | monetaria")
    for c in df.columns:
        partes.append(f"- {c} | {df[c].dtype} | {nulos_orig.get(c, int(df[c].isna().sum()))} | "
                      f"{df[c].nunique(dropna=True)} | {'sim' if eh_coluna_monetaria(c) else 'nao'}")

    if col_data:
        for c in col_data[:2]:
            s = df[c].dropna()
            if not s.empty:
                partes.append(f"\n[PERIODO] {c}: de {s.min():%d/%m/%Y} a {s.max():%d/%m/%Y}")

    if col_num:
        partes.append("\n[CORRELACOES MAIS FORTES] (Pearson; p<0,05 = estatisticamente significativa)")
        for p in ranking_correlacoes(df, col_num[:15], top=6):
            r2_par = p['r'] ** 2
            partes.append(f"- {p['var_1']} x {p['var_2']}: r={p['r']:.3f} ({p['interpretacao']}), "
                          f"R²={r2_par * 100:.1f}% ({r2_par:.3f}), p={p['p_valor']:.4f}, n={p['n']}")

    metrica = _escolher_metrica_principal(df, col_num)
    cat, eta2 = _escolher_categoria_principal(df, col_cat, metrica)
    if metrica and cat:
        res = resumir_por_categoria(df, cat, metrica, 'mean')
        partes.append(f"\n[RESUMO POR CATEGORIA] {metrica} por {cat} (eta²={eta2 * 100:.1f}% ({eta2:.3f}))" if eta2 is not None
                      else f"\n[RESUMO POR CATEGORIA] {metrica} por {cat}")
        for _, r in res.head(max_cats).iterrows():
            partes.append(f"- {r[cat]}: n={int(r['registros'])}, media={r['media']:.2f}, "
                          f"mediana={r['mediana']:.2f}, soma={r['soma']:.2f}, participacao_soma={r['participacao_soma_pct']:.1f}%")
    for c in [x for x in col_cat if x != cat][:3]:
        vc = df[c].astype(str).value_counts().head(5)
        partes.append(f"\n[FREQUENCIA] {c}: " + "; ".join(f"{k}={v} ({v / len(df) * 100:.1f}%)" for k, v in vc.items()))
    return "\n".join(partes)


def gerar_parecer_estatistico_local(df, tipo_analise="Automático (Geral)", pergunta_usuario=""):
    """
    Gera um relatório executivo completo e fiel aos dados diretamente a partir
    dos cálculos estatísticos do DataFrame (KPIs, correlações com significância,
    segmentos com tamanho de amostra, outliers, assimetria e tendência temporal).
    Garante funcionamento 100% mesmo sem internet ou cota de API externa.
    """
    n_linhas, n_cols = df.shape
    col_num = colunas_numericas_analiticas(df)
    col_cat = colunas_categoricas(df)
    col_data = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    nulos_orig = df.attrs.get('nulos_originais', {})
    total_nulos = int(sum(nulos_orig.get(c, 0) for c in df.columns))
    total_celulas = max(1, n_linhas * n_cols)
    duplicatas = int(df.attrs.get('duplicatas_removidas', 0))
    metrica = _escolher_metrica_principal(df, col_num)
    recomendacoes = []

    # 1. Tabela de KPIs
    ordem = ([c for c in col_num if eh_coluna_monetaria(c)] + [c for c in col_num if not eh_coluna_monetaria(c)])[:8]
    linhas_kpi = [
        "| Indicador / Métrica | Média | Mediana | Intervalo (Mín – Máx) | Coef. Variação (CV) | Valores válidos |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for c in ordem:
        try:
            est = calcular_estatisticas_resumo(df[c])
        except ValueError:
            continue
        pref = obter_prefixo_coluna(c)
        cv_txt = f"{formatar_percentual(est['coef_variacao'])} ({interpretar_cv(est['coef_variacao'])})" if est['cv_valido'] else "n/a*"
        linhas_kpi.append(
            f"| **{c}** | {formatar_valor(est['media'], prefixo=pref)} | {formatar_valor(est['mediana'], prefixo=pref)} | "
            f"{formatar_valor(est['minimo'], prefixo=pref)} – {formatar_valor(est['maximo'], prefixo=pref)} | {cv_txt} | "
            f"{formatar_inteiro(est['contagem'])} |"
        )
    tabela_kpis = "\n".join(linhas_kpi) if len(linhas_kpi) > 2 else "_Nenhuma métrica numérica disponível._"

    # 2. Correlações (com significância estatística)
    diag_correlacao = "A base possui menos de duas variáveis numéricas para análise de correlação."
    if len(col_num) >= 2:
        pares = ranking_correlacoes(df, col_num[:15], top=3)
        if pares:
            p0 = pares[0]
            sig = "estatisticamente significativa (p < 0,05)" if p0['significativa'] else "**não** significativa estatisticamente (p ≥ 0,05)"
            diag_correlacao = (
                f"A relação mais forte ocorre entre **`{p0['var_1']}`** e **`{p0['var_2']}`**: "
                f"r = {formatar_decimal(p0['r'], 3)} (R² = **{formatar_r2(p0['r'] ** 2)}** de variação compartilhada), "
                f"correlação **{p0['interpretacao']}**, {sig}, com {formatar_inteiro(p0['n'])} pares válidos."
            )
            if len(pares) > 1:
                outros = "; ".join(
                    f"`{p['var_1']}` × `{p['var_2']}` (r = {formatar_decimal(p['r'], 3)} | R² = {formatar_percentual((p['r'] ** 2) * 100, 1)}, {p['interpretacao']})"
                    for p in pares[1:]
                )
                diag_correlacao += f" Também se destacam: {outros}."
            diag_correlacao += " *Correlação indica associação, não necessariamente causa e efeito.*"
            if abs(p0['r']) >= 0.5 and p0['significativa']:
                recomendacoes.append(
                    f"**Investigar a relação `{p0['var_1']}` × `{p0['var_2']}`:** a associação é {p0['interpretacao']} "
                    f"(explica {formatar_percentual((p0['r'] ** 2) * 100, 1)} do comportamento). "
                    f"Valide com a área de negócio se há relação de causa antes de usá-la para metas ou previsões "
                    f"(a aba *Ciência de Dados → Regressão Bivariada* quantifica o efeito)."
                )
        else:
            diag_correlacao = "Não há pares de variáveis com observações suficientes para correlação confiável."

    # 3. Segmentos (com tamanho de amostra)
    diag_categoria = "Não há coluna categórica adequada (2 a 30 grupos) para segmentação."
    cat, eta2 = _escolher_categoria_principal(df, col_cat, metrica)
    if cat and metrica:
        min_reg = 5 if n_linhas >= 100 else 1
        res = resumir_por_categoria(df, cat, metrica, 'mean', min_registros=min_reg)
        res = res.dropna(subset=['media'])
        if len(res) >= 2:
            pref_p = obter_prefixo_coluna(metrica)
            top, low = res.iloc[0], res.iloc[-1]
            dif_pct = ((top['media'] - low['media']) / abs(low['media']) * 100) if low['media'] else np.nan
            diag_categoria = (
                f"Segmentando **`{metrica}`** por **`{cat}`**, o maior valor médio está em **{top[cat]}** "
                f"({formatar_valor(top['media'], prefixo=pref_p)}; n = {formatar_inteiro(top['registros'])}) e o menor em "
                f"**{low[cat]}** ({formatar_valor(low['media'], prefixo=pref_p)}; n = {formatar_inteiro(low['registros'])})"
                + (f", diferença de {formatar_percentual(dif_pct)}." if not pd.isna(dif_pct) else ".")
            )
            if eh_coluna_monetaria(metrica) and not pd.isna(res['participacao_soma_pct']).all():
                lider_soma = res.sort_values('soma', ascending=False).iloc[0]
                diag_categoria += (
                    f" Em volume total, **{lider_soma[cat]}** concentra {formatar_percentual(lider_soma['participacao_soma_pct'])} "
                    f"da soma de `{metrica}`."
                )
            if eta2 is not None:
                diag_categoria += (
                    f" A categoria explica **{formatar_r2(eta2)}** da variação de `{metrica}` (η²)"
                    + (" — efeito relevante." if eta2 >= 0.06 else " — efeito pequeno; os grupos são parecidos entre si.")
                )
            if min_reg > 1:
                diag_categoria += f" *(Grupos com menos de {min_reg} registros foram desconsiderados.)*"
            if eta2 is not None and eta2 >= 0.06:
                recomendacoes.append(
                    f"**Atuar no segmento {low[cat]} (`{cat}`):** apresenta a menor média de `{metrica}`. "
                    f"Use o *Filtro de Segmentação* para comparar seu perfil com o de **{top[cat]}** e identificar as causas."
                )

    # 4. Outliers e forma da distribuição
    diag_outliers = "Sem métrica numérica para avaliar outliers."
    if metrica:
        est = calcular_estatisticas_resumo(df[metrica])
        diag_outliers = (
            f"Em **`{metrica}`**, {formatar_inteiro(est['outliers_iqr_count'])} registros "
            f"({formatar_percentual(est['outliers_iqr_pct'], 2)}) estão fora da faixa típica de "
            f"{formatar_valor(est['limite_inferior'], nome_coluna=metrica)} a {formatar_valor(est['limite_superior'], nome_coluna=metrica)} "
            f"(critério de Tukey/IQR: {formatar_inteiro(est['outliers_superiores'])} acima e {formatar_inteiro(est['outliers_inferiores'])} abaixo). "
            f"A distribuição é {est['assimetria_texto']} (assimetria = {formatar_decimal(est['assimetria'], 2)})"
            + (", por isso a **mediana** representa melhor o valor típico do que a média." if abs(est['assimetria']) >= 1 else ".")
        )
        if est['outliers_iqr_pct'] >= 5:
            recomendacoes.append(
                f"**Revisar os {formatar_inteiro(est['outliers_iqr_count'])} registros atípicos de `{metrica}`:** "
                f"verifique se são erros de digitação/integração ou casos reais excepcionais antes de tomar decisões sobre médias."
            )

    # 5. Tendência temporal
    diag_tempo = ""
    if col_data and metrica:
        try:
            agg = 'sum' if eh_coluna_monetaria(metrica) else 'mean'
            rp = executar_previsao_temporal(df, col_data[0], metrica, n_periodos=1, freq='M', agregacao=agg)
            dh = rp['df_historico']
            direcao = "crescimento" if rp['coef'] > 0 else "queda"
            diag_tempo = (
                f"Considerando {rp['n_periodos_historicos']} meses de `{col_data[0]}` "
                f"({formatar_periodo(dh['data'].iloc[0])} a {formatar_periodo(dh['data'].iloc[-1])}), "
                f"`{metrica}` ({'soma' if agg == 'sum' else 'média'} mensal) apresenta tendência de **{direcao}** de "
                f"{formatar_valor(abs(rp['coef']), nome_coluna=metrica)} por mês "
                f"(poder explicativo **R² = {formatar_r2(rp['r2'])}**)"
                + (" — tendência **estatisticamente significativa**." if rp['tendencia_significativa']
                   else " — porém **não significativa** estatisticamente (oscilação pode ser aleatória).")
            )
            if rp['periodo_incompleto'] and not rp['periodo_incompleto'].get('mantido'):
                diag_tempo += " O último mês, incompleto, foi desconsiderado no cálculo."
            if rp['tendencia_significativa'] and rp['coef'] < 0:
                recomendacoes.append(
                    f"**Alerta de tendência de queda em `{metrica}`:** a redução mensal é consistente (R² = {formatar_r2(rp['r2'])}). "
                    f"Priorize um plano de ação e acompanhe a projeção na aba *Ciência de Dados*."
                )
        except Exception:
            diag_tempo = ""

    # 6. Qualidade dos dados
    pct_nulos = total_nulos / total_celulas * 100
    diag_qualidade = (
        f"{formatar_inteiro(total_nulos)} valores ausentes na origem ({formatar_percentual(pct_nulos)} das células) e "
        f"{formatar_inteiro(duplicatas)} linhas duplicadas removidas."
    )
    if df.attrs.get('imputar_numericos'):
        diag_qualidade += " Ausentes numéricos foram preenchidos pela mediana (pode reduzir a variabilidade real)."
    else:
        diag_qualidade += " Ausentes numéricos foram mantidos fora dos cálculos (não distorcem médias e correlações)."
    if pct_nulos >= 5:
        piores = sorted(((c, nulos_orig.get(c, 0)) for c in df.columns), key=lambda x: x[1], reverse=True)[:3]
        recomendacoes.append(
            "**Melhorar a qualidade da coleta:** as colunas com mais ausências são "
            + ", ".join(f"`{c}` ({formatar_inteiro(v)})" for c, v in piores if v) + "."
        )

    if not recomendacoes:
        recomendacoes.append(
            "**Monitoramento contínuo:** não foram encontrados desvios críticos. Defina metas com base na mediana "
            "e alertas para valores acima do 3º quartil (Q3) das métricas principais."
        )
    recomendacoes.append(
        "**Acompanhamento preditivo:** revise periodicamente a tendência e o poder explicativo (R² em %) na aba *Ciência de Dados & Regressão*."
    )

    foco_cabecalho = (
        f"**Foco Solicitado:** {tipo_analise}" + (f" | **Pergunta:** *{pergunta_usuario.strip()}*" if pergunta_usuario.strip() else "")
        if (pergunta_usuario.strip() or tipo_analise != "Automático (Geral)")
        else "Monitorar padrões estruturais, correlações críticas, desvios operacionais e oportunidades estratégicas."
    )
    aviso_pergunta = (
        "\n> ℹ️ O **Motor Analítico Nativo** calcula indicadores estatísticos, mas não interpreta perguntas em linguagem natural. "
        "Para uma resposta direta à sua pergunta, use **Groq** ou **Gemini** com uma chave de API.\n"
        if pergunta_usuario.strip() else ""
    )

    insights = [
        f"**Correlação entre Variáveis:** {diag_correlacao}",
        f"**Diferenças entre Segmentos:** {diag_categoria}",
        f"**Outliers e Distribuição:** {diag_outliers}",
    ]
    if diag_tempo:
        insights.append(f"**Tendência Temporal:** {diag_tempo}")
    insights.append(f"**Qualidade dos Dados:** {diag_qualidade}")

    texto_insights = "\n".join(f"{i}. {t}" for i, t in enumerate(insights, 1))
    texto_recs = "\n".join(f"{i}. {t}" for i, t in enumerate(recomendacoes, 1))

    relatorio = f"""# InnovaNexus - Relatório Executivo de Dados
*Parecer Prescritivo de Negócio, Diagnóstico de ETL e Visualização de Indicadores*

---

> **🎯 Objetivo Estratégico:** {foco_cabecalho}
{aviso_pergunta}
---

## 📊 1. Indicadores-Chave de Desempenho (KPIs)

Base analisada: **{formatar_inteiro(n_linhas)} registros** distribuídos em **{n_cols} colunas** auditadas via ETL.

{tabela_kpis}

<small>*CV = desvio padrão ÷ média. "n/a" quando a métrica tem valores negativos (CV não interpretável).*</small>

---

## 💡 2. Insights Estratégicos da Operação

{texto_insights}

---

## 🚀 3. Ações Prescritivas para a Gestão

{texto_recs}
"""
    return relatorio

# ---------------------------------------------------------
# INTEGRAÇÃO COM IA MULTIMOTOR (AUTO-DESCOBERTA DE MODELOS)
# ---------------------------------------------------------
REGRAS_FIDELIDADE = (
    "REGRAS OBRIGATÓRIAS DE FIDELIDADE AOS DADOS:\n"
    "1. Use SOMENTE números presentes nos dados abaixo; nunca invente valores, metas ou percentuais.\n"
    "2. Ao citar uma média por grupo, informe também o tamanho do grupo (n).\n"
    "3. Correlação não é causalidade: só afirme relação causal se os dados permitirem; cite r e p-valor.\n"
    "4. Considere correlações com p >= 0,05 como não significativas.\n"
    "5. Valores monetários em R$ apenas para colunas marcadas como 'monetaria=sim'.\n"
    "6. Use o padrão numérico brasileiro (1.234,56) e escreva em Português do Brasil.\n"
    "7. Ao citar o Coeficiente de Determinação (R²) ou poder explicativo, apresente SEMPRE em porcentagem ou no formato híbrido executivo, ex.: 97,1% (0,971).\n"
    "8. Se os dados não permitirem responder algo, diga claramente o que falta.\n\n"
)


def construir_prompt_insights(df_info_str, df_describe_str, df_head_str, tipo_analise="Automático (Geral)",
                              pergunta_usuario="", perfil_str=""):
    info_safe = df_info_str[:4000]
    desc_safe = df_describe_str[:5000]
    head_safe = df_head_str[:1500]
    perfil_safe = (perfil_str or "")[:6000]
    bloco_perfil = f"Perfil Estatístico Calculado (fonte confiável):\n{perfil_safe}\n\n" if perfil_safe else ""

    eh_sob_demanda = bool(pergunta_usuario and pergunta_usuario.strip()) or (tipo_analise and tipo_analise != "Automático (Geral)")

    dados = (
        "--- DADOS PARA ANÁLISE ---\n"
        f"Tipos de Colunas:\n{info_safe}\n\n"
        f"Resumo Estatístico:\n{desc_safe}\n\n"
        f"{bloco_perfil}"
        f"Amostra de Dados:\n{head_safe}\n"
    )

    if not eh_sob_demanda:
        return (
            "Atue como um Especialista em Analytics e Designer de Relatórios Executivos.\n"
            "Analise a estrutura de dados fornecida abaixo e gere um parecer estruturado "
            "estritamente em Português do Brasil de alto nível executivo e estratégico.\n\n"
            f"{REGRAS_FIDELIDADE}"
            "Siga exatamente a seguinte estrutura e estilo em Markdown:\n\n"
            "# InnovaNexus - Relatório Executivo de Dados\n"
            "*Parecer Prescritivo de Negócio, Diagnóstico de ETL e Visualização de Indicadores*\n\n"
            "---\n\n"
            "> **🎯 Objetivo Estratégico:** Identificar padrões críticos, correlações e oportunidades de melhoria "
            "a partir dos indicadores reais da base de dados.\n\n"
            "---\n\n"
            "## 📊 1. Indicadores-Chave de Desempenho (KPIs)\n\n"
            "Apresente uma tabela markdown com os KPIs calculados na base (média, mediana, mín, máx).\n\n"
            "---\n\n"
            "## 💡 2. Insights Estratégicos da Operação\n\n"
            "Destaque em subseções numeradas os 3 diagnósticos principais encontrados, cada um com o número que o comprova.\n\n"
            "---\n\n"
            "## 🚀 3. Ações Prescritivas para a Gestão\n\n"
            "Forneça 3 recomendações acionáveis, cada uma ligada a um diagnóstico acima.\n\n"
            f"{dados}"
        )
    else:
        foco_tipo = f"- **Foco da Análise Solicitada:** {tipo_analise}\n" if tipo_analise and tipo_analise != "Automático (Geral)" else ""
        foco_pergunta = f"- **Pergunta / Insight Específico Desejado:** {pergunta_usuario.strip()}\n" if pergunta_usuario and pergunta_usuario.strip() else ""

        return (
            "Atue como um Especialista em Ciência de Dados, Analytics e Inteligência de Negócios.\n"
            "O usuário solicitou uma ANÁLISE SOB DEMANDA específica sobre o dataset enviado.\n"
            "Sua resposta deve ser gerada estritamente a partir dos dados fornecidos e refletir com precisão o que foi solicitado.\n\n"
            f"{REGRAS_FIDELIDADE}"
            "### SOLICITAÇÃO DO USUÁRIO:\n"
            f"{foco_tipo}"
            f"{foco_pergunta}\n"
            "Siga exatamente a seguinte estrutura e estilo em Markdown:\n\n"
            f"# InnovaNexus - Análise Sob Demanda: {tipo_analise if tipo_analise != 'Automático (Geral)' else 'Consulta Personalizada'}\n"
            "*Relatório Analítico Específico baseado nos Dados do Dataset*\n\n"
            "---\n\n"
            "> **🎯 Objetivo da Solicitação:** Responder diretamente à questão/demanda do usuário com base nos dados reais.\n\n"
            "---\n\n"
            "## 📌 1. Resposta Direta & Diagnóstico da Pergunta Solicitada\n\n"
            "Responda diretamente com dados, métricas e cálculos extraídos da base fornecida.\n\n"
            "---\n\n"
            "## 📊 2. Evidências dos Dados & Indicadores Relevantes\n\n"
            "Apresente tabela markdown ou lista de indicadores que fundamentam a resposta.\n\n"
            "---\n\n"
            "## 🚀 3. Insights Estratégicos & Recomendações Acionáveis\n\n"
            "Forneça conclusões práticas e prescritivas a partir dos achados.\n\n"
            f"{dados}"
        )


def limpar_e_formatar_resposta_ia(texto):
    """
    Higieniza e garante a formatação Markdown adequada da resposta da IA:
    1. Remove blocos de raciocínio interno (<think>...</think>, <thought>...</thought>).
    2. Isola a partir do título '# InnovaNexus' caso o modelo tenha emitido rascunho prévio.
    3. Remove cercas ```markdown externas que impedem a renderização.
    4. Remove recuo excessivo de espaços no início das linhas (que o Markdown transforma
       indevidamente em blocos de código monoespaçados sem formatação).
    """
    if not texto or not isinstance(texto, str):
        return ""

    t = texto.strip()

    # 1. Remover tags explícitas de thinking (DeepSeek, Qwen, Gemini Thinking, etc.)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<thought>.*?</thought>", "", t, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in t.lower():
        t = re.split(r"</?think>", t, flags=re.IGNORECASE)[-1]

    # 2. Se o modelo escreveu rascunho em inglês antes do cabeçalho oficial '# InnovaNexus',
    # corta tudo que veio antes do cabeçalho Markdown
    match_titulo = re.search(r"(^|\n)\s*#\s+InnovaNexus", t)
    if match_titulo:
        t = t[match_titulo.start():].lstrip("\n")

    # 3. Remover cercas de código envolvendo todo o documento (```markdown ... ```)
    t = re.sub(r"^```(?:markdown|md)?\s*\n", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\n```\s*$", "", t.strip())

    # 4. Evitar que linhas com 4+ espaços de indentação virem bloco de código (<pre><code>)
    linhas_limpas = []
    em_bloco_codigo = False
    for linha in t.splitlines():
        if linha.strip().startswith("```"):
            em_bloco_codigo = not em_bloco_codigo
            linhas_limpas.append(linha.strip())
            continue
        if not em_bloco_codigo:
            # Remove indentação de 4+ espaços em listas, títulos, tabelas e parágrafos
            linha_sem_tab = linha.replace("\t", "  ")
            stripped = linha_sem_tab.lstrip(" ")
            n_espacos = len(linha_sem_tab) - len(stripped)
            if stripped.startswith(("#", "|", ">", "---", "###", "##")):
                linha_sem_tab = stripped
            elif stripped.startswith(("*", "-", "+")) or re.match(r"^\d+\.\s", stripped):
                # Mantém no máximo 2 espaços para sub-itens de lista (nunca 4+)
                linha_sem_tab = ("  " if n_espacos >= 2 else "") + stripped
            elif n_espacos >= 4:
                linha_sem_tab = stripped
            linhas_limpas.append(linha_sem_tab)
        else:
            linhas_limpas.append(linha)

    return "\n".join(linhas_limpas).strip()


def chamar_api_groq(key_groq, system_msg, prompt):
    """
    Descobre dinamicamente quais modelos estão liberados na conta Groq do usuário
    via REST API e executa a geração no primeiro modelo de texto compatível.
    """
    headers = {
        "Authorization": f"Bearer {key_groq}",
        "Content-Type": "application/json"
    }

    # 1. Descobrir modelos reais disponíveis para esta chave
    modelos_Preferidos = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "qwen/qwen3-32b",
        "deepseek-r1-distill-llama-70b",
        "compound-beta",
        "llama3-70b-8192",
        "llama3-8b-8192",
    ]
    modelos_para_testar = []

    try:
        r_models = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=15)
        if r_models.status_code == 401:
            return None, "❌ Chave Groq inválida ou expirada (HTTP 401). Verifique se a chave começa com 'gsk_' em https://console.groq.com/keys."
        if r_models.status_code == 200:
            dados_m = r_models.json().get("data", [])
            ids_conta = [
                m.get("id", "") for m in dados_m
                if m.get("id") and not any(ign in m.get("id", "").lower() for ign in [
                    "whisper", "tts", "playai", "guard", "safeguard", "embed", "vision-preview"
                ])
            ]
            for mp in modelos_Preferidos:
                if mp in ids_conta:
                    modelos_para_testar.append(mp)
            for id_c in ids_conta:
                if id_c not in modelos_para_testar:
                    modelos_para_testar.append(id_c)
    except Exception:
        pass

    if not modelos_para_testar:
        modelos_para_testar = modelos_Preferidos

    erros_modelos = []
    for mod in modelos_para_testar:
        payload = {
            "model": mod,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.5
        }
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45
            )
            if resp.status_code == 200:
                choices = resp.json().get("choices", [])
                if choices:
                    texto = choices[0].get("message", {}).get("content", "")
                    texto_limpo = limpar_e_formatar_resposta_ia(texto)
                    if texto_limpo:
                        return texto_limpo, None
            elif resp.status_code == 401:
                return None, "❌ Chave Groq inválida (HTTP 401). Gere uma nova chave 'gsk_...' em https://console.groq.com/keys."
            else:
                try:
                    det = resp.json().get("error", {}).get("message", resp.text[:150])
                except Exception:
                    det = resp.text[:150]
                erros_modelos.append(f"{mod}: {det}")
                continue
        except Exception as e_req:
            erros_modelos.append(f"{mod}: {str(e_req)}")
            continue

    resumo_erros = " | ".join(erros_modelos[:3])
    return None, (
        f"❌ Não foi possível gerar resposta nos modelos da Groq ({resumo_erros}).\n\n"
        "💡 **Como verificar no Console da Groq:** Acesse `https://console.groq.com/settings/limits` e verifique se os modelos Llama estão habilitados no seu projeto."
    )


def chamar_api_gemini(key_gemini, system_msg, prompt):
    """
    Descobre dinamicamente os modelos Gemini habilitados para a chave do usuário
    via endpoint REST oficial v1beta/models e executa generateContent.
    Filtra partes de raciocínio interno ('thought': True) e garante tokens suficientes.
    """
    headers = {"Content-Type": "application/json"}
    # Prioriza modelos diretos e rápidos antes de modelos thinking pesados
    modelos_preferidos = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-flash-latest",
        "gemini-2.5-pro",
        "gemini-pro-latest",
    ]
    modelos_para_testar = []

    # 1. Consultar a lista oficial de modelos que a chave do usuário tem acesso
    try:
        url_lista = f"https://generativelanguage.googleapis.com/v1beta/models?key={key_gemini}"
        r_list = requests.get(url_lista, timeout=15)
        if r_list.status_code in (400, 403):
            try:
                msg_err = r_list.json().get("error", {}).get("message", r_list.text[:200])
            except Exception:
                msg_err = r_list.text[:200]
            return None, f"❌ Chave Gemini recusada pela Google (HTTP {r_list.status_code}): {msg_err}"
        if r_list.status_code == 200:
            lista_m = r_list.json().get("models", [])
            disponiveis = []
            for m in lista_m:
                metodos = m.get("supportedGenerationMethods", [])
                nome_m = m.get("name", "").replace("models/", "")
                if (
                    "generateContent" in metodos
                    and not any(ign in nome_m.lower() for ign in ("tts", "image", "embedding", "thinking", "vision", "audio"))
                ):
                    disponiveis.append(nome_m)
            for mp in modelos_preferidos:
                if mp in disponiveis:
                    modelos_para_testar.append(mp)
            for disp in disponiveis:
                if disp not in modelos_para_testar:
                    modelos_para_testar.append(disp)
    except Exception:
        pass

    if not modelos_para_testar:
        modelos_para_testar = modelos_preferidos

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_msg}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "maxOutputTokens": 8192,
            "temperature": 0.5
        }
    }

    erros_gemini = []
    for mod in modelos_para_testar:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={key_gemini}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                dados = resp.json()
                candidatos = dados.get("candidates", [])
                if candidatos:
                    partes = candidatos[0].get("content", {}).get("parts", [])
                    # Ignora partes de raciocínio interno ("thought": True) e concatena apenas a resposta final
                    textos_finais = [
                        p.get("text", "")
                        for p in partes
                        if p.get("text") and not p.get("thought", False)
                    ]
                    if not textos_finais and partes:
                        # Fallback: pega a última parte se todas vieram sem flag explícita
                        textos_finais = [partes[-1].get("text", "")]
                    texto_bruto = "\n".join(textos_finais).strip()
                    texto_limpo = limpar_e_formatar_resposta_ia(texto_bruto)
                    if texto_limpo and len(texto_limpo) > 80:
                        return texto_limpo, None
            elif resp.status_code == 429:
                erros_gemini.append(f"{mod}: Limite de cota atingido (429)")
                continue
            else:
                try:
                    det = resp.json().get("error", {}).get("message", resp.text[:150])
                except Exception:
                    det = resp.text[:150]
                erros_gemini.append(f"{mod} (HTTP {resp.status_code}): {det}")
                continue
        except Exception as e_req:
            erros_gemini.append(f"{mod}: {str(e_req)}")
            continue

    resumo_erros = " | ".join(erros_gemini[:3])
    return None, f"❌ Falha na API Gemini ({resumo_erros}). Verifique sua chave em https://aistudio.google.com/apikey."


def gerar_insights_ia(motor, api_key, df_info_str, df_describe_str, df_head_str, tipo_analise="Automático (Geral)", pergunta_usuario="", df_ref=None, perfil_str=""):
    """
    Roteador inteligente de IA com auto-detecção de chave (gsk_ vs AIza) e
    auto-descoberta de modelos ativos na conta do usuário.
    """
    try:
        if motor == "Motor Analítico Nativo (Offline / Sem Chave)" and df_ref is not None:
            return gerar_parecer_estatistico_local(df_ref, tipo_analise, pergunta_usuario), None

        chave = (api_key or "").strip()
        if not chave:
            return None, f"🔑 Chave de API não informada para **{motor}**. Insira a chave na barra lateral ou selecione o **Motor Analítico Nativo**."

        prompt = construir_prompt_insights(df_info_str, df_describe_str, df_head_str, tipo_analise, pergunta_usuario, perfil_str)
        system_msg = (
            "Você é um analista de dados sênior especialista em business intelligence e ciência de dados. "
            "Você é rigoroso: só afirma o que os números fornecidos sustentam. "
            "Responda DIRETAMENTE com o relatório final formatado em Markdown (Português do Brasil), "
            "começando imediatamente pelo título '# InnovaNexus', sem exibir rascunho ou raciocínio prévio em inglês "
            "e sem identar o texto com 4 espaços."
        )

        # Auto-detecção inteligente caso o usuário tenha colado chave Gemini no seletor Groq ou vice-versa
        if chave.startswith("AIza") or chave.startswith("AQ."):
            motor_efetivo = "Gemini (Google)"
        elif chave.startswith("gsk_"):
            motor_efetivo = "Groq (Rápido/Gratuito)"
        else:
            motor_efetivo = motor

        if motor_efetivo == "Groq (Rápido/Gratuito)":
            return chamar_api_groq(chave, system_msg, prompt)
        elif motor_efetivo == "Gemini (Google)":
            return chamar_api_gemini(chave, system_msg, prompt)

        return None, "Motor de IA não reconhecido."

    except Exception as e:
        return None, f"Erro crítico na função de IA: {str(e)}"

# ---------------------------------------------------------
# TELA DE LOGIN
# ---------------------------------------------------------
if not st.session_state['autenticado']:
    col_centered = st.columns([1, 2, 1])
    with col_centered[1]:
        st.markdown("## 🚀 InnovaNexus")
        st.subheader("Portal de Acesso Executivo")
        with st.form("form_login"):
            usuario_input = st.text_input("Utilizador", value="admin")
            senha_input = st.text_input("Palavra-passe", type="password", value="1234")
            btn_entrar = st.form_submit_button("Entrar na Plataforma")
            if btn_entrar:
                realizar_login(usuario_input, senha_input)

# ---------------------------------------------------------
# APLICAÇÃO PRINCIPAL
# ---------------------------------------------------------
else:
    st.title("🚀 InnovaNexus: Insights & Dashboards com IA")

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    st.sidebar.header("👤 Sessão do Utilizador")
    st.sidebar.write("Conectado como: **Administrador**")
    st.sidebar.markdown("---")

    st.sidebar.header("🤖 Motor de Inteligência Artificial")
    motor_selecionado = st.sidebar.selectbox(
        "Selecione o serviço de IA:",
        [
            "Groq (Rápido/Gratuito)",
            "Gemini (Google)",
            "Motor Analítico Nativo (Offline / Sem Chave)"
        ]
    )

    if motor_selecionado != "Motor Analítico Nativo (Offline / Sem Chave)":
        chave_personalizada = st.sidebar.text_input(
            "Chave de API",
            type="password",
            help="Cole aqui sua chave da Groq (começa com gsk_) ou do Gemini (começa com AIza). O sistema reconhece automaticamente o provedor."
        )
        chave_ativa = chave_personalizada.strip() if chave_personalizada.strip() else CHAVES_PADRAO.get(motor_selecionado, "")
        chave_valida = bool(chave_ativa)

        # Aviso inteligente se colou a chave trocada
        if chave_ativa.startswith("AIza") and motor_selecionado == "Groq (Rápido/Gratuito)":
            st.sidebar.info("🔄 Chave **Gemini (`AIza...`)** detectada! O sistema usará a API do Google Gemini automaticamente.")
        elif chave_ativa.startswith("gsk_") and motor_selecionado == "Gemini (Google)":
            st.sidebar.info("🔄 Chave **Groq (`gsk_...`)** detectada! O sistema usará a API da Groq automaticamente.")

        if chave_valida:
            st.sidebar.success("✅ Chave de API Pronta")
        else:
            st.sidebar.warning(f"⚠️ Insira a chave de API ou escolha o Motor Analítico Nativo.")

        links_chave = {
            "Groq (Rápido/Gratuito)": "https://console.groq.com/keys",
            "Gemini (Google)":        "https://aistudio.google.com/apikey",
        }
        if motor_selecionado in links_chave:
            st.sidebar.caption(f"🔑 [Obter chave {motor_selecionado.split()[0]}]({links_chave[motor_selecionado]})")
    else:
        chave_ativa = "NATIVO"
        chave_valida = True
        st.sidebar.success("✅ Motor Analítico Nativo Ativo (Não requer chave externa)")

    # ── FONTE DE DADOS ─────────────────────────────────────────────────────────
    st.markdown("---")
    c_fonte, c_etl = st.columns([2, 1])
    with c_fonte:
        fonte_dados = st.radio(
            "Escolha a fonte de entrada de dados:",
            ["Upload de Arquivo (.csv)", "Conexão com Banco de Dados (SQL)"],
            horizontal=True
        )
    with c_etl:
        with st.expander("⚙️ Opções de Tratamento (ETL)"):
            remover_dup_opt = st.checkbox("Remover linhas duplicadas", value=True)
            imputar_num_opt = st.checkbox(
                "Preencher ausentes numéricos com a mediana",
                value=False,
                help="Desmarcado (recomendado): valores ausentes são ignorados nos cálculos, preservando a variabilidade e as correlações reais."
            )

    df_base = None
    id_contexto = ""

    if fonte_dados == "Upload de Arquivo (.csv)":
        arquivo_carregado = st.file_uploader(
            "Selecione um ficheiro CSV para análise",
            type=["csv"],
            key="uploader_csv"
        )
        if arquivo_carregado is not None:
            try:
                bruto_bytes = arquivo_carregado.getvalue()
                id_contexto = f"csv:{hashlib.md5(bruto_bytes).hexdigest()[:12]}:{remover_dup_opt}:{imputar_num_opt}"
                df_raw = ler_csv_robusto(arquivo_carregado)
                df_base = executar_etl_automatico(df_raw, imputar_numericos=imputar_num_opt, remover_duplicatas=remover_dup_opt)
                dup_txt = f" | **{formatar_inteiro(df_base.attrs.get('duplicatas_removidas', 0))} duplicadas removidas**" if df_base.attrs.get('duplicatas_removidas', 0) else ""
                st.success(
                    f"✅ Ficheiro carregado e processado via ETL! "
                    f"**{formatar_inteiro(df_base.shape[0])} registros** × **{df_base.shape[1]} colunas**{dup_txt}."
                )
            except Exception as e:
                st.error(f"Erro ao ler o ficheiro CSV: {e}")

    elif fonte_dados == "Conexão com Banco de Dados (SQL)":
        st.subheader("🔌 Conexão com Banco de Dados SQL")
        col1, col2 = st.columns(2)
        with col1:
            db_type = st.selectbox("Tipo de Banco de Dados", ["PostgreSQL", "MySQL", "SQLite"])
            db_host = st.text_input("Servidor / Host", value="localhost")
            db_port = st.text_input("Porta de Conexão", value="5432")
        with col2:
            db_user = st.text_input("Utilizador do Banco")
            db_pass = st.text_input("Palavra-passe", type="password")
            db_name = st.text_input("Nome do Banco / Arquivo SQLite")

        db_table = st.text_input("Nome da Tabela para Consulta")
        btn_conectar = st.button("Conectar e Carregar Tabela")

        if btn_conectar:
            if db_table:
                with st.spinner("A ligar ao banco de dados..."):
                    df_sql_raw, erro = carregar_dados_sql(db_type, db_host, db_port, db_user, db_pass, db_name, db_table)
                    if erro:
                        st.error(f"Erro na conexão: {erro}")
                    else:
                        st.session_state['df_sql_raw'] = df_sql_raw
                        st.session_state['sql_id'] = f"sql:{db_type}:{db_name}:{db_table}:{len(df_sql_raw)}"
                        st.success("✅ Dados carregados com sucesso a partir da base de dados!")
            else:
                st.warning("Por favor, informe o nome da tabela.")

        if 'df_sql_raw' in st.session_state:
            df_base = executar_etl_automatico(
                st.session_state['df_sql_raw'],
                imputar_numericos=imputar_num_opt,
                remover_duplicatas=remover_dup_opt
            )
            id_contexto = f"{st.session_state.get('sql_id', 'sql')}:{remover_dup_opt}:{imputar_num_opt}"

    # ── FILTRO DINÂMICO DE SEGMENTAÇÃO NA SIDEBAR ──────────────────────────────
    df = df_base
    resumo_filtro = "Todos os registros (sem filtro)"
    if df_base is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Filtro de Segmentação")
        col_cat_filtro = [c for c in colunas_categoricas(df_base) if 2 <= df_base[c].nunique(dropna=True) <= 30]
        if col_cat_filtro:
            col_filtro_sel = st.sidebar.selectbox(
                "Filtrar base por categoria:",
                ["(Todos os Registros)"] + col_cat_filtro
            )
            if col_filtro_sel != "(Todos os Registros)":
                valores_disp = sorted(df_base[col_filtro_sel].astype(str).unique().tolist())
                valores_sel = st.sidebar.multiselect(
                    f"Valores de '{col_filtro_sel}':",
                    valores_disp,
                    default=valores_disp
                )
                if valores_sel:
                    df = df_base[df_base[col_filtro_sel].astype(str).isin(valores_sel)].copy()
                    df.attrs = dict(df_base.attrs)
                    id_contexto += f"|{col_filtro_sel}={','.join(valores_sel)}"
                    if len(valores_sel) < len(valores_disp):
                        resumo_filtro = f"`{col_filtro_sel}` em ({', '.join(valores_sel)}) — {formatar_inteiro(len(df))} de {formatar_inteiro(len(df_base))} registros"
                    st.sidebar.caption(f"🔎 Exibindo **{formatar_inteiro(len(df))}** de **{formatar_inteiro(len(df_base))}** registros.")
                else:
                    st.sidebar.warning("Selecione pelo menos um valor para aplicar o filtro.")

    # Limpa resultados em cache quando o dataset ou o filtro mudam (evita exibir análises antigas)
    if id_contexto and st.session_state.get('_id_contexto') != id_contexto:
        for chave_antiga in ('insights_ia', 'res_prev_temp', 'res_biv'):
            st.session_state.pop(chave_antiga, None)
        st.session_state['_id_contexto'] = id_contexto

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Sair (Logout)"):
        realizar_logout()

    # ══════════════════════════════════════════════════════════════════════════
    # DASHBOARDS — só renderiza se há dados carregados
    # ══════════════════════════════════════════════════════════════════════════
    if df is not None and not df.empty:
        st.markdown("---")
        if resumo_filtro != "Todos os registros (sem filtro)":
            st.info(f"🔎 **Filtro ativo:** {resumo_filtro}. Todas as abas, gráficos e modelos refletem este recorte.")

        todas_num = [c for c in df.select_dtypes(include=['number']).columns if not pd.api.types.is_bool_dtype(df[c])]
        col_num = colunas_numericas_analiticas(df)
        col_cat = colunas_categoricas(df) or colunas_texto(df)
        col_data = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        nulos_orig_map = df.attrs.get('nulos_originais', {})
        if resumo_filtro == "Todos os registros (sem filtro)" and nulos_orig_map:
            nulos_orig_series = pd.Series([nulos_orig_map.get(c, 0) for c in df.columns], index=df.columns)
        else:
            # Com filtro ativo, calcula os ausentes do próprio recorte
            nulos_orig_series = pd.Series([
                int((df[c] == "Não informado").sum()) if c in colunas_texto(df) else int(df[c].isna().sum())
                for c in df.columns
            ], index=df.columns)
        total_nulos_tratados = int(nulos_orig_series.sum())
        nulos_atuais = df.isna().sum()

        info_df = pd.DataFrame({
            "Coluna":              df.columns,
            "Tipo de Dado":        [str(d) for d in df.dtypes],
            "Ausentes na Origem":  nulos_orig_series.values,
            "% Ausentes (Origem)": (nulos_orig_series.values / max(1, len(df)) * 100).round(1),
            "Ausentes Atuais":     nulos_atuais.values,
            "Valores Únicos":      [df[c].nunique(dropna=True) for c in df.columns],
            "Unidade":             ["Monetária (R$)" if eh_coluna_monetaria(c) else "Geral" for c in df.columns],
        })

        tab_dados, tab_diagnostico, tab_ia, tab_dash, tab_ciencia_dados = st.tabs([
            "👀 Visão Geral dos Dados",
            "🔬 Diagnóstico Técnico",
            "🤖 Parecer da IA & Insights",
            "📊 Dashboard Interativo",
            "📈 Ciência de Dados & Regressão"
        ])

        # ── TAB 1: VISÃO GERAL ─────────────────────────────────────────────────
        with tab_dados:
            st.subheader("📋 Visão Geral dos Dados")

            total_celulas = max(1, df.shape[0] * df.shape[1])
            completude = max(0.0, (1 - total_nulos_tratados / total_celulas) * 100)

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("📦 Registros Ativos",     formatar_inteiro(df.shape[0]))
            k2.metric("📐 Colunas",              f"{df.shape[1]}")
            k3.metric("🔢 Métricas Numéricas",   f"{len(col_num)}")
            k4.metric("🏷️ Colunas Categóricas", f"{len(col_cat)}")
            k5.metric("✅ Completude da Origem", formatar_percentual(completude),
                      help=f"{formatar_inteiro(total_nulos_tratados)} células ausentes na origem.")

            if col_num:
                st.markdown("##### 📌 Resumo Rápido das Principais Métricas")
                cols_resumo = ([c for c in col_num if eh_coluna_monetaria(c)] + [c for c in col_num if not eh_coluna_monetaria(c)])[:4]
                cr_cols = st.columns(len(cols_resumo))
                for col_ui, nome_c in zip(cr_cols, cols_resumo):
                    s_c = pd.to_numeric(df[nome_c], errors='coerce').dropna()
                    if len(s_c) == 0:
                        continue
                    pref_c = obter_prefixo_coluna(nome_c)
                    if eh_coluna_monetaria(nome_c):
                        col_ui.metric(f"💰 Soma de {nome_c}", formatar_valor(s_c.sum(), prefixo=pref_c),
                                      f"Média: {formatar_valor(s_c.mean(), prefixo=pref_c)}", delta_color="off")
                    else:
                        col_ui.metric(f"📊 Média de {nome_c}", formatar_valor(s_c.mean(), prefixo=pref_c),
                                      f"Mediana: {formatar_valor(s_c.median(), prefixo=pref_c)}", delta_color="off")

            st.markdown("---")

            max_slider = min(500, max(10, df.shape[0]))
            n_linhas = st.slider(
                "Número de linhas a exibir:",
                min_value=min(10, max_slider),
                max_value=max_slider,
                value=min(50, max_slider),
                step=10
            ) if df.shape[0] > 10 else df.shape[0]
            st.dataframe(df.head(n_linhas), use_container_width=True, height=500)

            csv_tratado = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📥 Baixar Dataset Tratado (CSV)",
                data=csv_tratado,
                file_name="dataset_tratado.csv",
                mime="text/csv"
            )

        # ── TAB 2: DIAGNÓSTICO TÉCNICO ─────────────────────────────────────────
        with tab_diagnostico:
            st.subheader("🔬 Diagnóstico Técnico de Qualidade & Auditoria de ETL")

            dup_rem = int(df.attrs.get('duplicatas_removidas', 0))
            convs = df.attrs.get('conversoes', [])
            st.caption(
                f"**Auditoria do ETL:** {formatar_inteiro(dup_rem)} duplicatas removidas · "
                f"{formatar_inteiro(total_nulos_tratados)} ausentes identificados na origem · "
                f"Imputação numérica: **{'Ativa (mediana)' if df.attrs.get('imputar_numericos') else 'Desativada (preserva distribuição real)'}**"
                + (f" · Conversões automáticas: {', '.join(convs[:5])}" if convs else "")
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 📋 Estrutura e Qualidade por Coluna")
                st.dataframe(info_df, use_container_width=True, height=400, hide_index=True)

            with c2:
                if col_num:
                    st.markdown("##### 📈 Estatísticas Descritivas (Variáveis Numéricas)")
                    desc = df[col_num].describe().T.rename(columns={
                        'count': 'Válidos', 'mean': 'Média', 'std': 'Desv. Padrão',
                        'min': 'Mínimo', '25%': 'Q1 (25%)', '50%': 'Mediana', '75%': 'Q3 (75%)', 'max': 'Máximo'
                    })
                    desc['CV (%)'] = [
                        round(float(df[c].std() / abs(df[c].mean()) * 100), 1)
                        if df[c].dropna().mean() != 0 and (df[c].dropna() >= 0).all() else np.nan
                        for c in col_num
                    ]
                    desc['Assimetria'] = [df[c].skew() if df[c].dropna().nunique() > 2 else 0.0 for c in col_num]
                    st.dataframe(desc.round(2), use_container_width=True, height=400)

            if total_nulos_tratados > 0:
                st.markdown("##### 🛠️ Valores Ausentes na Origem por Coluna")
                ausentes = pd.DataFrame({
                    'Coluna': nulos_orig_series.index,
                    'Ausentes': nulos_orig_series.values,
                    '% da Base': (nulos_orig_series.values / max(1, len(df)) * 100).round(1)
                })
                ausentes = ausentes[ausentes['Ausentes'] > 0].sort_values('Ausentes', ascending=False)
                fig_missing = px.bar(
                    ausentes, x='Coluna', y='Ausentes',
                    hover_data=['% da Base'],
                    title="Colunas com Valores Ausentes na Origem",
                    template="plotly_dark",
                    color='Ausentes',
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                st.plotly_chart(fig_missing, use_container_width=True)

            if len(col_num) >= 2:
                st.markdown("##### 🔗 Matriz de Correlação entre Variáveis Numéricas")
                corr = df[col_num].corr().round(2)
                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    template="plotly_dark",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="Correlação Linear de Pearson (-1 a +1)",
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                pares_top = ranking_correlacoes(df, col_num[:15], top=8)
                if pares_top:
                    st.markdown("##### 🏆 Pares de Variáveis Mais Correlacionados (com Poder Explicativo R²)")
                    df_pares = pd.DataFrame([{
                        "Variável 1": p["var_1"],
                        "Variável 2": p["var_2"],
                        "Correlação (r)": f"{formatar_percentual(p['r'] * 100, 1, sinal=True)} ({formatar_decimal(p['r'], 3)})",
                        "Poder Explicativo (R²)": formatar_r2(p["r"] ** 2),
                        "Interpretação": p["interpretacao"].capitalize(),
                        "p-valor": f"{p['p_valor']:.4f}",
                        "Significativa (p < 0,05)": "Sim ✅" if p["significativa"] else "Não ⚠️",
                        "Pares Válidos (n)": formatar_inteiro(p["n"]),
                    } for p in pares_top])
                    st.dataframe(df_pares, use_container_width=True, hide_index=True)

        # ── TAB 3: IA & INSIGHTS ───────────────────────────────────────────────
        with tab_ia:
            st.subheader(f"🧠 Parecer & Insights com IA ({motor_selecionado.split()[0]})")

            st.markdown("##### ⚙️ Configuração da Análise")
            c_modo, c_tipo = st.columns([1, 1])
            with c_modo:
                modo_analise = st.radio(
                    "Modo de Análise:",
                    ["Automático (Padrão)", "Sob Demanda (Personalizado)"],
                    horizontal=True,
                    help="No modo Automático, a IA gera o parecer executivo geral completo. No modo Sob Demanda, você especifica o tipo de análise e perguntas pontuais sobre o dataset."
                )

            tipo_analise_input = "Automático (Geral)"
            pergunta_usuario_input = ""

            if modo_analise == "Sob Demanda (Personalizado)":
                with c_tipo:
                    tipo_analise_input = st.selectbox(
                        "Tipo de Análise Solicitada:",
                        [
                            "Diagnóstico Operacional & Gargalos",
                            "Impacto Cognitivo & Saúde Mental",
                            "Previsão & Tendências de Futuro",
                            "Financeiro, Custos & Rentabilidade",
                            "Desempenho e Produtividade",
                            "Comportamento e Segmentação de Perfis",
                            "Qualidade e Conformidade dos Dados",
                            "Personalizado / Outro"
                        ]
                    )
                pergunta_usuario_input = st.text_area(
                    "Pergunta ou informação específica desejada sobre o dataset:",
                    placeholder="Ex: Qual o impacto do aumento das horas de uso de IA na capacidade de pensamento crítico e no tempo de foco?",
                    help="Informe a pergunta, indicador ou foco desejado. A análise refletirá estritamente o que foi solicitado com base no dataset enviado."
                )
            else:
                with c_tipo:
                    st.info("💡 **Modo Automático Ativo:** Análise executiva geral gerada automaticamente sobre a estrutura do dataset.")

            st.markdown("---")

            btn_texto = "✨ Gerar Parecer Executivo Automático" if modo_analise == "Automático (Padrão)" else "🎯 Gerar Análise Sob Demanda"
            if st.button(btn_texto, type="primary"):
                with st.spinner(f"A processar dados via {motor_selecionado}... Aguarde."):
                    info_str     = info_df.to_string(index=False)
                    describe_str = df[col_num].describe().T.to_string() if col_num else "Nenhuma variável numérica."
                    head_str     = df.head(5).to_string(index=False)
                    perfil_str   = construir_perfil_dados(df, col_num, col_cat, col_data)

                    insights, erro_ia = gerar_insights_ia(
                        motor_selecionado, chave_ativa, info_str, describe_str, head_str,
                        tipo_analise=tipo_analise_input,
                        pergunta_usuario=pergunta_usuario_input,
                        df_ref=df,
                        perfil_str=perfil_str
                    )

                    if erro_ia:
                        st.warning(
                            f"⚠️ A API externa retornou um aviso ({erro_ia})\n\n"
                            "✅ **Contingência Inteligente Ativada:** O parecer abaixo foi gerado automaticamente pelo **Motor Analítico Nativo do InnovaNexus** diretamente a partir das estatísticas do seu dataset!"
                        )
                        st.session_state['insights_ia'] = gerar_parecer_estatistico_local(
                            df, tipo_analise_input, pergunta_usuario_input
                        )
                    else:
                        st.session_state['insights_ia'] = insights

            if 'insights_ia' in st.session_state:
                texto_exibir = limpar_e_formatar_resposta_ia(st.session_state['insights_ia'])
                st.markdown(texto_exibir)

                if col_num and col_cat:
                    try:
                        var_num_pdf = st.session_state.get('dash_var_num', _escolher_metrica_principal(df, col_num))
                        var_cat_pdf = st.session_state.get('dash_var_cat', _escolher_categoria_principal(df, col_cat, var_num_pdf)[0])
                        if var_num_pdf in df.columns and var_cat_pdf in df.columns:
                            pdf_data = gerar_pdf_executivo(
                                df.shape[0], df.shape[1],
                                total_nulos_tratados,
                                st.session_state['insights_ia'],
                                df, var_num_pdf, var_cat_pdf
                            )
                            st.download_button(
                                "📥 Baixar Relatório PDF Executivo",
                                data=pdf_data,
                                file_name="InnovaNexus_Relatorio.pdf",
                                mime="application/pdf"
                            )
                    except Exception as e:
                        st.warning(f"PDF não gerado: {e}")

        # ── TAB 4: DASHBOARD INTERATIVO ────────────────────────────────────────
        with tab_dash:
            st.subheader("📊 Dashboard Interativo")

            if not col_num:
                st.info("Nenhuma coluna numérica encontrada para gerar gráficos.")
            elif not col_cat:
                st.info("Nenhuma coluna categórica encontrada para gerar gráficos.")
            else:
                col_sel1, col_sel2, col_sel3, col_sel4 = st.columns(4)
                var_num = col_sel1.selectbox("📈 Métrica principal (Eixo Y):", col_num, key="dash_var_num")
                var_cat = col_sel2.selectbox("🏷️ Categoria (Eixo X):", col_cat, key="dash_var_cat")
                agg_padrao = 1 if eh_coluna_monetaria(var_num) else 0
                agg_dash_label = col_sel3.selectbox(
                    "∑ Agregação por categoria:",
                    ["Média", "Soma", "Mediana", "Contagem de Registros"],
                    index=agg_padrao
                )
                agg_dash = {"Média": "mean", "Soma": "sum", "Mediana": "median", "Contagem de Registros": "count"}[agg_dash_label]
                var_num2 = col_sel4.selectbox(
                    "📉 2ª Métrica (comparativo):",
                    [c for c in col_num if c != var_num] or col_num
                )

                c_sl1, c_sl2 = st.columns(2)
                top_n = c_sl1.slider("Top N categorias:", min_value=5, max_value=30, value=10, step=5)
                min_reg_cat = c_sl2.slider(
                    "Mínimo de registros por categoria:",
                    min_value=1, max_value=max(2, min(50, len(df) // 5 or 2)), value=1,
                    help="Evita que categorias com apenas 1 ou 2 registros distorçam o ranking de médias."
                )

                df_cat_full = resumir_por_categoria(df, var_cat, var_num, agregacao=agg_dash, min_registros=min_reg_cat)
                df_top = df_cat_full.head(top_n).copy()
                pref_dash = obter_prefixo_coluna(var_num) if agg_dash != "count" else ""
                df_top["valor_fmt"] = df_top["valor"].apply(
                    lambda v: formatar_inteiro(v) if agg_dash == "count" else formatar_valor(v, prefixo=pref_dash)
                )

                # Se a categoria for mês ("01 - Janeiro"), ordena cronologicamente no eixo X
                eh_mes_cat = var_cat.lower().endswith("_mes")
                if eh_mes_cat:
                    df_top = df_top.sort_values(var_cat)

                st.markdown("---")

                # Linha 1: Barras + Pizza (com proporção matematicamente válida)
                r1c1, r1c2 = st.columns([3, 2])
                with r1c1:
                    titulo_bar = (
                        f"📅 {agg_dash_label} de '{var_num}' por Mês" if eh_mes_cat
                        else f"🏆 Top {len(df_top)}: {agg_dash_label} de '{var_num}' por '{var_cat}'"
                    )
                    fig_bar = px.bar(
                        df_top, x=var_cat, y="valor",
                        text="valor_fmt",
                        custom_data=["registros", "media", "mediana", "soma"],
                        title=titulo_bar,
                        labels={"valor": f"{agg_dash_label} de {var_num}"},
                        template="plotly_dark",
                        color="valor",
                        color_continuous_scale="Blues"
                    )
                    fig_bar.update_traces(
                        textposition="outside",
                        hovertemplate=(
                            f"<b>%{{x}}</b><br>{agg_dash_label}: %{{text}}<br>"
                            "Registros (n): %{customdata[0]}<br>Média: %{customdata[1]:,.2f}<br>"
                            "Mediana: %{customdata[2]:,.2f}<br>Soma: %{customdata[3]:,.2f}<extra></extra>"
                        )
                    )
                    fig_bar.update_layout(xaxis_tickangle=-35, showlegend=False, uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig_bar, use_container_width=True)

                with r1c2:
                    # Pizza e Treemap representam PARTES DE UM TODO: usam Soma (quando positiva e faz sentido) ou Contagem de registros
                    soma_positiva = (pd.to_numeric(df[var_num], errors='coerce').dropna() >= 0).all() and df_top["soma"].sum() > 0
                    usar_soma_prop = soma_positiva and (eh_coluna_monetaria(var_num) or agg_dash == "sum")
                    col_prop = "soma" if usar_soma_prop else "registros"
                    titulo_pie = (
                        f"🥧 Participação na Soma de '{var_num}' por '{var_cat}'"
                        if usar_soma_prop else f"🥧 Participação no Total de Registros por '{var_cat}'"
                    )
                    fig_pie = px.pie(
                        df_top, values=col_prop, names=var_cat,
                        title=titulo_pie,
                        template="plotly_dark",
                        hole=0.35
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    if agg_dash in ("mean", "median"):
                        st.caption("ℹ️ Gráficos de pizza exigem partes de um todo; por isso exibem a participação nos registros/soma, e não a média.")

                # Linha 2: Box Plot + Histograma
                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    cats_top = df_top[var_cat].tolist()
                    df_box = df[df[var_cat].astype(str).isin(cats_top)]
                    fig_box = px.box(
                        df_box, x=var_cat, y=var_num,
                        category_orders={var_cat: cats_top},
                        title=f"📦 Dispersão e Mediana: '{var_num}' por '{var_cat}'",
                        template="plotly_dark",
                        color=var_cat,
                        notched=False
                    )
                    fig_box.update_layout(xaxis_tickangle=-35, showlegend=False)
                    st.plotly_chart(fig_box, use_container_width=True)

                with r2c2:
                    s_hist = pd.to_numeric(df[var_num], errors='coerce').dropna()
                    fig_hist = px.histogram(
                        df, x=var_num,
                        title=f"📊 Distribuição de '{var_num}' (Média vs Mediana)",
                        template="plotly_dark",
                        nbins=40,
                        color_discrete_sequence=["#3b82f6"],
                        marginal="box"
                    )
                    if len(s_hist) > 0:
                        fig_hist.add_vline(x=float(s_hist.mean()), line_dash="dash", line_color="#ef4444",
                                           annotation_text=f"Média: {formatar_valor(s_hist.mean(), nome_coluna=var_num)}",
                                           annotation_position="top right")
                        fig_hist.add_vline(x=float(s_hist.median()), line_dash="dot", line_color="#10b981",
                                           annotation_text=f"Mediana: {formatar_valor(s_hist.median(), nome_coluna=var_num)}",
                                           annotation_position="top left")
                    fig_hist.update_layout(bargap=0.05)
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Linha 3: Scatter + Comparativo (com eixo duplo quando as escalas são diferentes)
                r3c1, r3c2 = st.columns(2)
                with r3c1:
                    if len(col_num) >= 2 and var_num != var_num2:
                        try:
                            import statsmodels  # noqa: F401
                            _trendline = "ols"
                        except ImportError:
                            _trendline = None

                        corr_sc = calcular_correlacao(df[var_num], df[var_num2])
                        r_txt = (
                            f" (R² = {formatar_percentual((corr_sc['r'] ** 2) * 100, 1)} | r = {formatar_decimal(corr_sc['r'], 2)} — {corr_sc['interpretacao']})"
                            if not pd.isna(corr_sc['r']) else ""
                        )
                        df_sc_sample = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df
                        fig_scatter = px.scatter(
                            df_sc_sample, x=var_num, y=var_num2,
                            color=var_cat if df[var_cat].nunique() <= 12 else None,
                            title=f"🔵 Dispersão: '{var_num}' vs '{var_num2}'{r_txt}",
                            template="plotly_dark",
                            trendline=_trendline,
                            opacity=0.7
                        )
                        fig_scatter.update_layout(showlegend=df[var_cat].nunique() <= 8)
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("Scatter plot requer ao menos 2 colunas numéricas distintas.")

                with r3c2:
                    if len(col_num) >= 2 and var_num2 in df.columns and var_num != var_num2:
                        df_comp = (
                            df[df[var_cat].astype(str).isin(cats_top)]
                            .assign(**{var_cat: df[var_cat].astype(str)})
                            .groupby(var_cat)[[var_num, var_num2]]
                            .mean()
                            .reindex(cats_top)
                            .reset_index()
                        )
                        m1_abs = float(df_comp[var_num].abs().max() or 1.0)
                        m2_abs = float(df_comp[var_num2].abs().max() or 1.0)
                        razao_escala = max(m1_abs, m2_abs) / max(1e-9, min(m1_abs, m2_abs))

                        if razao_escala > 5:
                            # Escalas incompatíveis (ex.: R$ 50.000 vs 8 horas): usa 2 eixos independentes
                            fig_comp = go.Figure()
                            fig_comp.add_trace(go.Bar(
                                x=df_comp[var_cat], y=df_comp[var_num],
                                name=f"Média de {var_num}", marker_color="#3b82f6", yaxis="y1"
                            ))
                            fig_comp.add_trace(go.Scatter(
                                x=df_comp[var_cat], y=df_comp[var_num2],
                                name=f"Média de {var_num2}", mode="lines+markers",
                                line=dict(color="#f59e0b", width=3), marker=dict(size=9), yaxis="y2"
                            ))
                            fig_comp.update_layout(
                                title=f"📊 Comparativo em Dupla Escala: '{var_num}' (barras) vs '{var_num2}' (linha)",
                                xaxis=dict(tickangle=-35),
                                yaxis=dict(title=var_num),
                                yaxis2=dict(title=var_num2, overlaying="y", side="right", showgrid=False),
                                template="plotly_dark",
                                legend=dict(orientation="h", y=1.12)
                            )
                        else:
                            df_melt = df_comp.melt(
                                id_vars=var_cat, value_vars=[var_num, var_num2],
                                var_name="Métrica", value_name="Média"
                            )
                            fig_comp = px.bar(
                                df_melt, y=var_cat, x="Média", color="Métrica",
                                orientation='h',
                                title=f"📊 Comparativo de Médias: '{var_num}' vs '{var_num2}'",
                                template="plotly_dark",
                                barmode="group"
                            )
                        st.plotly_chart(fig_comp, use_container_width=True)
                    else:
                        fig_hbar = px.bar(
                            df_top.sort_values("valor"),
                            y=var_cat, x="valor",
                            orientation='h',
                            title=f"📊 Ranking Horizontal: {agg_dash_label} de '{var_num}'",
                            template="plotly_dark",
                            color="valor",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig_hbar, use_container_width=True)

                # Linha 4: Série Temporal (agregada por período) + Treemap
                r4c1, r4c2 = st.columns(2)
                with r4c1:
                    if col_data:
                        col_dt = col_data[0]
                        s_dt = df[col_dt].dropna()
                        dias_span = (s_dt.max() - s_dt.min()).days if len(s_dt) >= 2 else 0
                        freq_ts = "Y" if dias_span > 365 * 4 else ("M" if dias_span > 60 else "D")
                        rotulo_freq = {"Y": "Anual", "M": "Mensal", "D": "Diária"}[freq_ts]
                        df_ts_base = pd.DataFrame({"dt": df[col_dt], "val": pd.to_numeric(df[var_num], errors='coerce')}).dropna()
                        df_ts_base["periodo"] = df_ts_base["dt"].dt.to_period(freq_ts).dt.to_timestamp()
                        func_ts = "sum" if agg_dash == "sum" else ("median" if agg_dash == "median" else "mean")
                        rot_ts = "Soma" if func_ts == "sum" else ("Mediana" if func_ts == "median" else "Média")
                        df_ts = df_ts_base.groupby("periodo")["val"].agg(valor=func_ts, n="size").reset_index().sort_values("periodo")
                        fig_ts = px.line(
                            df_ts, x="periodo", y="valor",
                            hover_data={"n": True},
                            title=f"📅 Evolução {rotulo_freq} ({rot_ts} de '{var_num}' em '{col_dt}')",
                            labels={"periodo": "Período", "valor": f"{rot_ts} de {var_num}", "n": "Registros no período"},
                            template="plotly_dark",
                            markers=True
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)
                    else:
                        cats_top2 = df_top[var_cat].tolist()
                        df_vio = df[df[var_cat].astype(str).isin(cats_top2)]
                        fig_vio = px.violin(
                            df_vio, x=var_cat, y=var_num,
                            box=True, points="outliers",
                            title=f"🎻 Densidade (Violin Plot): '{var_num}' por '{var_cat}'",
                            template="plotly_dark",
                            color=var_cat
                        )
                        fig_vio.update_layout(showlegend=False, xaxis_tickangle=-35)
                        st.plotly_chart(fig_vio, use_container_width=True)

                with r4c2:
                    if len(col_num) >= 1 and len(col_cat) >= 1 and not df_top.empty:
                        df_tree = df_cat_full.head(min(20, len(df_cat_full))).copy()
                        fig_tree = px.treemap(
                            df_tree,
                            path=[var_cat],
                            values=col_prop,
                            color="media",
                            hover_data={"registros": True, "media": ":.2f", "mediana": ":.2f", "soma": ":.2f"},
                            title=f"🌳 Treemap: Tamanho = {'Soma' if usar_soma_prop else 'Registros'} | Cor = Média de '{var_num}'",
                            template="plotly_dark",
                            color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig_tree, use_container_width=True)

                st.markdown("---")
                st.markdown(f"##### 📋 Tabela de Resumo Estatístico de '{var_num}' por '{var_cat}'")
                df_resumo = df.groupby(df[var_cat].astype(str))[var_num].agg(
                    Registros='count',
                    Soma='sum',
                    Média='mean',
                    Mediana='median',
                    DesvPad='std',
                    Mínimo='min',
                    Máximo='max'
                ).reset_index()
                df_resumo['% dos Registros'] = (df_resumo['Registros'] / max(1, df_resumo['Registros'].sum()) * 100).round(1)
                if (df_resumo['Soma'] > 0).all() and df_resumo['Soma'].sum() > 0:
                    df_resumo['% da Soma'] = (df_resumo['Soma'] / df_resumo['Soma'].sum() * 100).round(1)
                df_resumo = df_resumo[df_resumo['Registros'] >= min_reg_cat].round(2).sort_values(
                    'Soma' if agg_dash == 'sum' else ('Mediana' if agg_dash == 'median' else ('Registros' if agg_dash == 'count' else 'Média')),
                    ascending=False
                )
                st.dataframe(df_resumo, use_container_width=True, height=350, hide_index=True)

        # ── TAB 5: CIÊNCIA DE DADOS & REGRESSÃO ───────────────────────────────
        with tab_ciencia_dados:
            st.subheader("📈 Ciência de Dados: Regressão Linear & Estatística Avançada")
            st.markdown(
                "Módulo preditivo e estatístico integrado ao InnovaNexus. "
                "Execute modelagem de **Regressão Linear** para projeções futuras temporais (com intervalo de previsão de 95%) "
                "ou análise de relação bivariada, além de diagnósticos detalhados de **Quartis, Outliers, Assimetria e Variabilidade**."
            )

            if not col_num:
                st.warning("⚠️ O dataset atual não contém colunas numéricas suficientes para regressão e estatística avançada.")
            else:
                sub_reg, sub_estat = st.tabs([
                    "🔮 Regressão Linear & Projeções",
                    "📊 Estatísticas Avançadas & Diagnóstico Preditivo"
                ])

                # ── SUB-ABA 1: REGRESSÃO LINEAR & PROJEÇÕES ───────────────────
                with sub_reg:
                    tipo_reg = st.radio(
                        "Tipo de Modelagem:",
                        [
                            "Projeção Temporal (Tendência e Previsão Futura)",
                            "Regressão Bivariada (Relação entre X e Y)"
                        ],
                        horizontal=True
                    )

                    if tipo_reg == "Projeção Temporal (Tendência e Previsão Futura)":
                        col_datas_possiveis = col_data + [
                            c for c in df.columns
                            if c not in col_data and _parece_coluna_data(c)
                        ] + [c for c in df.columns if c.lower() in ('year', 'ano') or c.lower().endswith('_ano')]
                        col_datas_possiveis = list(dict.fromkeys(col_datas_possiveis))

                        if not col_datas_possiveis:
                            st.info("ℹ️ Nenhuma coluna de data identificada automaticamente. Selecione a coluna temporal desejada:")
                            col_datas_possiveis = df.columns.tolist()

                        c_reg1, c_reg2, c_reg3, c_reg4 = st.columns(4)
                        with c_reg1:
                            col_data_sel = st.selectbox("📅 Coluna Temporal (Data/Ano):", col_datas_possiveis)
                        with c_reg2:
                            col_valor_sel = st.selectbox("📈 Métrica a Projetar (Y):", col_num)

                        is_moeda_sel = eh_coluna_monetaria(col_valor_sel)
                        eh_ano_inteiro = col_data_sel.lower() in ('year', 'ano', 'ano_num') or col_data_sel.lower().endswith('_ano')

                        with c_reg3:
                            opcoes_freq = ["Anual", "Mensal", "Trimestral", "Diário"] if eh_ano_inteiro else ["Mensal", "Trimestral", "Anual", "Diário"]
                            freq_sel = st.selectbox("⏱️ Periodicidade:", opcoes_freq)
                            freq_map = {"Mensal": "M", "Diário": "D", "Trimestral": "Q", "Anual": "Y"}
                        with c_reg4:
                            default_agg_idx = 0 if is_moeda_sel else 1
                            agg_sel = st.selectbox("∑ Agregação:", ["Soma", "Média", "Mediana"], index=default_agg_idx)
                            agg_val = {"Soma": "sum", "Média": "mean", "Mediana": "median"}[agg_sel]

                        c_p1, c_p2, c_p3 = st.columns([2, 1, 1])
                        with c_p1:
                            n_periodos_sel = st.slider(
                                "Quantos períodos futuros deseja prever?",
                                min_value=1,
                                max_value=24,
                                value=6,
                                help="Número de passos futuros projetados com base na tendência linear dos dados históricos."
                            )
                        with c_p2:
                            ignorar_parcial = st.checkbox(
                                "Descartar último período se incompleto",
                                value=True,
                                help="Quando a agregação é Soma, um último mês/ano com poucos dias puxa a reta de tendência artificialmente para baixo."
                            )
                        with c_p3:
                            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
                            btn_gerar_prev = st.button("🔮 Gerar Previsão Temporal", type="primary")

                        st.caption(f"🏷️ Unidade identificada automaticamente: **{'Monetária (R$)' if is_moeda_sel else 'Quantidade / Numérica Geral'}**")

                        if btn_gerar_prev:
                            try:
                                with st.spinner("Treinando modelo de regressão linear e calculando projeções..."):
                                    res_prev = executar_previsao_temporal(
                                        df, col_data_sel, col_valor_sel,
                                        n_periodos=n_periodos_sel,
                                        freq=freq_map[freq_sel],
                                        agregacao=agg_val,
                                        remover_periodo_incompleto=ignorar_parcial
                                    )
                                    res_prev['freq_label'] = freq_sel
                                    st.session_state['res_prev_temp'] = res_prev
                            except Exception as e:
                                st.error(f"❌ Não foi possível executar a regressão temporal: {e}")

                        if 'res_prev_temp' in st.session_state:
                            rp = st.session_state['res_prev_temp']
                            df_hist = rp['df_historico']
                            df_prev = rp['df_previsoes']
                            is_moeda_res = eh_coluna_monetaria(rp['col_valor'])
                            prefixo_res = "R$ " if is_moeda_res else ""
                            freq_cod = rp.get('freq', 'M')

                            if rp.get('periodo_incompleto'):
                                pi = rp['periodo_incompleto']
                                per_txt = formatar_periodo(pi['periodo'], freq_cod)
                                if pi.get('mantido'):
                                    st.warning(
                                        f"⚠️ O último período (**{per_txt}**) parece incompleto "
                                        f"(cobre cerca de {formatar_percentual(pi['cobertura_pct'], 0)} do período) e foi mantido no ajuste."
                                    )
                                else:
                                    st.info(
                                        f"ℹ️ O último período (**{per_txt}**, valor {formatar_valor(pi['valor'], prefixo=prefixo_res)}) "
                                        f"estava incompleto ({formatar_percentual(pi['cobertura_pct'], 0)} do período) e foi removido para não distorcer a soma."
                                    )
                            if rp.get('periodos_sem_dados', 0) > 0:
                                st.info(
                                    f"ℹ️ Foram identificados **{rp['periodos_sem_dados']} período(s) intermediário(s) sem registros**; "
                                    f"o modelo preservou a distância real no calendário."
                                )
                            if rp.get('valores_negativos_previstos'):
                                st.warning(
                                    "⚠️ A tendência linear de queda projeta valores **negativos** em períodos futuros para uma métrica "
                                    "historicamente positiva. Interprete como tendência de redução, não como valor literal negativo."
                                )

                            st.markdown("---")
                            st.markdown("#### 📈 Qualidade do Ajuste do Modelo (Regressão Linear)")

                            unidade_tempo = {
                                "Y": "ano", "M": "mês", "Q": "trimestre", "D": "dia", "W": "semana"
                            }.get(freq_cod, "período")
                            media_hist_val = rp.get('media_historica', float(df_hist[rp['col_valor']].mean()))
                            mae_pct = rp.get('mae_pct_media', (abs(rp['mae'] / media_hist_val) * 100 if media_hist_val else np.nan))
                            rmse_pct = rp.get('rmse_pct_media', (abs(rp['rmse'] / media_hist_val) * 100 if media_hist_val else np.nan))
                            coef_pct = rp.get('coef_pct_media', ((rp['coef'] / abs(media_hist_val)) * 100 if media_hist_val else np.nan))

                            mae_fmt = formatar_valor(rp['mae'], prefixo=prefixo_res)
                            rmse_fmt = formatar_valor(rp['rmse'], prefixo=prefixo_res)
                            media_hist_fmt = formatar_valor(media_hist_val, prefixo=prefixo_res)
                            coef_fmt = formatar_valor(rp['coef'], prefixo=prefixo_res)
                            sinal_coef = "+" if rp['coef'] > 0 else ""

                            mae_card_val = f"{mae_fmt} (±{formatar_percentual(mae_pct, 1)})" if (not pd.isna(mae_pct) and len(mae_fmt) <= 8) else mae_fmt
                            rmse_card_val = f"{rmse_fmt} (±{formatar_percentual(rmse_pct, 1)})" if (not pd.isna(rmse_pct) and len(rmse_fmt) <= 8) else rmse_fmt

                            m1, m2, m3, m4 = st.columns(4)
                            with m1:
                                m1.metric(
                                    "R² (Determinação)",
                                    formatar_r2(rp['r2']),
                                    help="Coeficiente de Determinação (R²): indica a porcentagem do comportamento da métrica que é explicada pelo modelo linear (0% a 100%)."
                                )
                                sig_txt = "Tendência significativa (p < 0,05) ✅" if rp.get('tendencia_significativa') else "Tendência não significativa (p ≥ 0,05) ⚠️"
                                m1.caption(f"{formatar_percentual(rp['r2'] * 100, 1)} do comportamento explicado · {sig_txt}")
                            with m2:
                                m2.metric(
                                    "MAE (Erro Médio Absoluto)",
                                    mae_card_val,
                                    help="Erro Médio Absoluto: desvio médio entre os valores reais e a linha de tendência, na mesma unidade da métrica e em % da média histórica."
                                )
                                if not pd.isna(mae_pct):
                                    m2.caption(f"Margem de erro média de ±{formatar_percentual(mae_pct, 1)} sobre a média histórica ({media_hist_fmt})")
                                else:
                                    m2.caption("Erro médio absoluto nas unidades da métrica")
                            with m3:
                                m3.metric(
                                    "RMSE (Raiz do Erro Médio²)",
                                    rmse_card_val,
                                    help="Raiz do Erro Quadrático Médio: dá peso maior a grandes desvios. Quando próximo ao MAE, indica que os erros são regulares e sem picos extremos."
                                )
                                rmse_diag = (
                                    "Erros regulares e sem picos extremos (próximo ao MAE)"
                                    if (rp['mae'] > 0 and (rp['rmse'] / rp['mae']) <= 1.35)
                                    else "Sensível a picos de desvio (há oscilações maiores na série)"
                                )
                                m3.caption(rmse_diag)
                            with m4:
                                m4.metric(
                                    f"Coeficiente Angular (por {unidade_tempo.capitalize()})",
                                    f"{sinal_coef}{coef_fmt} / {unidade_tempo}",
                                    help=f"Inclinação da reta de regressão: representa o ritmo médio de variação a cada {unidade_tempo}."
                                )
                                dir_coef = "Queda média" if rp['coef'] < 0 else ("Crescimento médio" if rp['coef'] > 0 else "Estabilidade")
                                pct_coef_txt = f" ({formatar_percentual(coef_pct, 1, sinal=True)}/{unidade_tempo} sobre a média)" if not pd.isna(coef_pct) else ""
                                m4.caption(f"{dir_coef} de {formatar_valor(abs(rp['coef']), prefixo=prefixo_res)} por {unidade_tempo}{pct_coef_txt}")

                            st.markdown("#### 📊 Histórico, Tendência e Intervalo de Previsão (95%)")
                            fig_prev = go.Figure()

                            # Faixa de incerteza (IC de previsão de 95%)
                            fig_prev.add_trace(go.Scatter(
                                x=pd.concat([df_prev['data'], df_prev['data'][::-1]]),
                                y=pd.concat([df_prev['limite_superior'], df_prev['limite_inferior'][::-1]]),
                                fill='toself',
                                fillcolor='rgba(16, 185, 129, 0.18)',
                                line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo='skip',
                                name='Intervalo de Previsão (95%)'
                            ))

                            fig_prev.add_trace(go.Scatter(
                                x=df_hist['data'],
                                y=df_hist[rp['col_valor']],
                                mode='lines+markers',
                                name='Histórico Observado',
                                line=dict(color='#3b82f6', width=3),
                                marker=dict(size=7, color='#3b82f6')
                            ))

                            fig_prev.add_trace(go.Scatter(
                                x=df_hist['data'],
                                y=df_hist['tendencia'],
                                mode='lines',
                                name='Linha de Tendência (Regressão)',
                                line=dict(color='#f59e0b', width=2, dash='dash')
                            ))

                            fig_prev.add_trace(go.Scatter(
                                x=df_prev['data'],
                                y=df_prev['valor_previsto'],
                                mode='lines+markers',
                                name=f'Projeção Futura ({rp["n_periodos"]} períodos)',
                                line=dict(color='#10b981', width=3),
                                marker=dict(size=9, color='#10b981', symbol='diamond')
                            ))

                            fig_prev.add_trace(go.Scatter(
                                x=[df_hist['data'].iloc[-1], df_prev['data'].iloc[0]],
                                y=[df_hist[rp['col_valor']].iloc[-1], df_prev['valor_previsto'].iloc[0]],
                                mode='lines',
                                name='Conexão',
                                line=dict(color='#10b981', width=2, dash='dot'),
                                showlegend=False
                            ))

                            fig_prev.update_layout(
                                title=(
                                    f"Análise de Tendência e Previsão de '{rp['col_valor']}' "
                                    f"({rp['n_periodos_historicos']} períodos históricos | R² = {formatar_r2(rp['r2'])})"
                                ),
                                xaxis_title="Período",
                                yaxis_title=rp['col_valor'],
                                template="plotly_dark",
                                height=500,
                                hovermode="x unified"
                            )
                            st.plotly_chart(fig_prev, use_container_width=True)

                            c_tab, c_bar = st.columns([1, 1])

                            with c_tab:
                                col_prev_label = f"Valor Previsto ({prefixo_res.strip()})" if prefixo_res else "Valor Previsto"
                                st.markdown("##### 📋 Tabela de Projeção Detalhada")
                                df_prev_view = df_prev.copy()
                                df_prev_view['periodo_fmt'] = df_prev_view['data'].apply(lambda d: formatar_periodo(d, freq_cod))
                                df_prev_view['valor_fmt'] = df_prev_view['valor_previsto'].apply(
                                    lambda v: formatar_valor(v, prefixo=prefixo_res)
                                )
                                df_prev_view['faixa_fmt'] = [
                                    f"{formatar_valor(li, prefixo=prefixo_res)} a {formatar_valor(ls, prefixo=prefixo_res)}"
                                    for li, ls in zip(df_prev_view['limite_inferior'], df_prev_view['limite_superior'])
                                ]
                                df_prev_view['crescimento_fmt'] = df_prev_view['crescimento_pct'].apply(
                                    lambda x: formatar_percentual(x, 2, sinal=True) if not pd.isna(x) else "-"
                                )
                                st.dataframe(
                                    df_prev_view[['periodo_fmt', 'valor_fmt', 'faixa_fmt', 'crescimento_fmt']].rename(columns={
                                        'periodo_fmt': 'Período',
                                        'valor_fmt': col_prev_label,
                                        'faixa_fmt': 'Intervalo 95%',
                                        'crescimento_fmt': 'Variação (%)'
                                    }),
                                    use_container_width=True,
                                    height=320,
                                    hide_index=True
                                )

                            with c_bar:
                                st.markdown("##### 📊 Comparativo: Recente vs Previsão")
                                n_hist_comp = min(6, len(df_hist))
                                df_hist_comp = df_hist.tail(n_hist_comp)

                                fig_comp = go.Figure()
                                fig_comp.add_trace(go.Bar(
                                    x=[formatar_periodo(d, freq_cod) for d in df_hist_comp['data']],
                                    y=df_hist_comp[rp['col_valor']],
                                    name='Histórico Recente',
                                    marker_color='#3b82f6'
                                ))
                                fig_comp.add_trace(go.Bar(
                                    x=[formatar_periodo(d, freq_cod) for d in df_prev['data']],
                                    y=df_prev['valor_previsto'],
                                    name='Previsão Futura',
                                    marker_color='#10b981'
                                ))
                                fig_comp.update_layout(
                                    title=f"Últimos {n_hist_comp} Períodos vs Previsão",
                                    xaxis_title="Período",
                                    yaxis_title=rp['col_valor'],
                                    barmode="group",
                                    template="plotly_dark",
                                    height=320
                                )
                                st.plotly_chart(fig_comp, use_container_width=True)

                            st.markdown("##### 📌 Resumo Executivo da Previsão")
                            ult_per_hist_txt = formatar_periodo(df_hist['data'].iloc[-1], freq_cod)
                            ult_per_prev_txt = formatar_periodo(df_prev['data'].iloc[-1], freq_cod)
                            ult_val_prev_fmt = formatar_valor(df_prev['valor_previsto'].iloc[-1], prefixo=prefixo_res)

                            r1, r2, r3, r4 = st.columns(4)
                            with r1:
                                if rp.get('agregacao') == 'sum':
                                    r1.metric("Soma / Acumulado Previsto", formatar_valor(rp['total_previsto'], prefixo=prefixo_res))
                                    r1.caption(f"Total projetado nos próximos {rp['n_periodos']} períodos")
                                else:
                                    r1.metric(
                                        f"Faixa Prevista ({ult_per_prev_txt})",
                                        f"{formatar_valor(df_prev['limite_inferior'].iloc[-1], prefixo=prefixo_res)} – {formatar_valor(df_prev['limite_superior'].iloc[-1], prefixo=prefixo_res)}"
                                    )
                                    r1.caption(f"Intervalo de confiança de 95% no fim do horizonte ({ult_per_prev_txt})")
                            with r2:
                                r2.metric("Média Periódica Prevista", formatar_valor(rp['media_mensal'], prefixo=prefixo_res))
                                r2.caption(f"Média projetada vs {media_hist_fmt} na série histórica")
                            with r3:
                                r3.metric(
                                    "Variação Esperada (Fim vs Histórico)",
                                    formatar_percentual(rp['crescimento_total'], 2, sinal=True),
                                    f"Pela tendência: {formatar_percentual(rp.get('crescimento_tendencia', 0), 2, sinal=True)}",
                                    delta_color="off"
                                )
                                r3.caption(f"De {ult_per_hist_txt} até {ult_per_prev_txt}")
                            with r4:
                                r4.metric("Último Ponto Histórico", formatar_valor(rp['ultimo_historico'], prefixo=prefixo_res))
                                r4.caption(f"Real em {ult_per_hist_txt} · Previsto em {ult_per_prev_txt}: {ult_val_prev_fmt}")

                            with st.expander("ℹ️ Detalhes Matemáticos e Interpretação Executiva do Modelo", expanded=True):
                                p_txt = f"{rp['p_valor_coef']:.4f}" if not pd.isna(rp.get('p_valor_coef')) else "N/A"
                                st.markdown(f"""
                                - **Poder Explicativo ($R^2$):** **{formatar_r2(rp['r2'])}** — comunica de imediato que **{formatar_percentual(rp['r2'] * 100, 1)}** do comportamento de `{rp['col_valor']}` ao longo do tempo é explicado pelo modelo linear ({interpretar_r2(rp['r2'])}).
                                - **Ritmo de Variação (Coeficiente Angular):** **{sinal_coef}{coef_fmt} por {unidade_tempo}**{f" ({formatar_percentual(coef_pct, 1, sinal=True)}/{unidade_tempo} em relação à média histórica)" if not pd.isna(coef_pct) else ""} · p-valor = `{p_txt}`.
                                - **Precisão das Estimativas (MAE e RMSE):** O modelo erra em média **±{mae_fmt}**{f" (apenas **{formatar_percentual(mae_pct, 1)}** da média histórica de {media_hist_fmt})" if not pd.isna(mae_pct) else ""}, com RMSE de **{rmse_fmt}**{f" (±{formatar_percentual(rmse_pct, 1)})" if not pd.isna(rmse_pct) else ""}.
                                - **Equação da Reta:** `Y = {formatar_decimal(rp['intercept'])} + ({formatar_decimal(rp['coef'])}) × X` *(onde X é o número de períodos desde {formatar_periodo(df_hist['data'].iloc[0], freq_cod)})*.
                                - **Avaliação do Ajuste:** {'✅ **Ajuste linear forte (R² ≥ 70%):** a tendência histórica é altamente consistente e confiável para projeções.' if rp['r2'] >= 0.7 else 'ℹ️ **Ajuste moderado (50% ≤ R² < 70%):** há tendência clara, mas com oscilações; utilize a faixa do Intervalo de Previsão (95%) para tomada de decisão.' if rp['r2'] >= 0.5 else '⚠️ **Ajuste fraco (R² < 50%):** os dados oscilam fortemente ao redor da reta. A previsão indica apenas a direção geral; filtre um segmento específico na barra lateral para maior precisão.'}
                                """)

                    else:
                        st.markdown("##### 🔬 Regressão Linear entre Duas Variáveis Numéricas")
                        col_bx, col_by = st.columns(2)
                        with col_bx:
                            x_col_sel = st.selectbox("Variável Explicativa / Independente (X):", col_num, index=0)
                        with col_by:
                            y_opts = [c for c in col_num if c != x_col_sel] or col_num
                            y_col_sel = st.selectbox("Variável Resposta / Dependente (Y):", y_opts, index=0)

                        if st.button("🔬 Ajustar Modelo Linear (X → Y)", type="primary"):
                            try:
                                with st.spinner("Ajustando modelo de regressão linear..."):
                                    res_biv = executar_regressao_bivariada(df, x_col_sel, y_col_sel)
                                    st.session_state['res_biv'] = res_biv
                            except Exception as e:
                                st.error(f"❌ Erro na regressão bivariada: {e}")

                        if 'res_biv' in st.session_state:
                            rb = st.session_state['res_biv']
                            df_biv = rb['df_clean']
                            prefixo_rb = obter_prefixo_coluna(rb['col_y'])
                            prefixo_rx = obter_prefixo_coluna(rb['col_x'])

                            media_y_biv = rb.get('media_y', float(df_biv[rb['col_y']].mean()))
                            mae_biv_pct = rb.get('mae_pct_media', (abs(rb['mae'] / media_y_biv) * 100 if media_y_biv else np.nan))
                            rmse_biv_pct = rb.get('rmse_pct_media', (abs(rb['rmse'] / media_y_biv) * 100 if media_y_biv else np.nan))
                            mae_biv_fmt = formatar_valor(rb['mae'], prefixo=prefixo_rb)
                            rmse_biv_fmt = formatar_valor(rb['rmse'], prefixo=prefixo_rb)

                            mae_biv_card = f"{mae_biv_fmt} (±{formatar_percentual(mae_biv_pct, 1)})" if (not pd.isna(mae_biv_pct) and len(mae_biv_fmt) <= 8) else mae_biv_fmt
                            rmse_biv_card = f"{rmse_biv_fmt} (±{formatar_percentual(rmse_biv_pct, 1)})" if (not pd.isna(rmse_biv_pct) and len(rmse_biv_fmt) <= 8) else rmse_biv_fmt

                            st.markdown("---")
                            b1, b2, b3, b4 = st.columns(4)
                            with b1:
                                b1.metric(
                                    "R² (Determinação)",
                                    formatar_r2(rb['r2']),
                                    help="Porcentagem da variação de Y que é explicada diretamente pela variável X."
                                )
                                b1.caption(
                                    f"Explica {formatar_percentual(rb['r2'] * 100, 1)} de Y · "
                                    f"r = {formatar_percentual(rb.get('r_pearson', 0) * 100, 1, sinal=True)} ({formatar_decimal(rb.get('r_pearson', 0), 3)} — {rb.get('interpretacao', '')})"
                                )
                            with b2:
                                b2.metric("MAE (Erro Médio Absoluto)", mae_biv_card)
                                b2.caption(
                                    f"Erro médio de ±{formatar_percentual(mae_biv_pct, 1)} sobre a média de Y"
                                    if not pd.isna(mae_biv_pct) else "Erro médio absoluto nas unidades de Y"
                                )
                            with b3:
                                b3.metric("RMSE (Raiz do Erro Médio²)", rmse_biv_card)
                                b3.caption(
                                    "Erros regulares sem picos extremos"
                                    if (rb['mae'] > 0 and (rb['rmse'] / rb['mae']) <= 1.35)
                                    else "Penaliza desvios maiores na relação X → Y"
                                )
                            with b4:
                                sinal_biv = "+" if rb['coef'] > 0 else ""
                                b4.metric("Inclinação (ΔY por +1 em X)", f"{sinal_biv}{formatar_decimal(rb['coef'], 4)}")
                                b4.caption("Significativa (p < 0,05) ✅" if rb.get('significativa') else "Não significativa (p ≥ 0,05) ⚠️")

                            df_biv_plot = df_biv.sample(min(3000, len(df_biv)), random_state=42) if len(df_biv) > 3000 else df_biv
                            fig_biv = px.scatter(
                                df_biv_plot, x=rb['col_x'], y=rb['col_y'],
                                title=f"Dispersão com Linha de Regressão: {rb['col_y']} vs {rb['col_x']} (R² = {formatar_r2(rb['r2'])})",
                                template="plotly_dark",
                                opacity=0.65
                            )
                            df_sorted = df_biv.sort_values(rb['col_x'])
                            fig_biv.add_trace(go.Scatter(
                                x=df_sorted[rb['col_x']],
                                y=df_sorted['y_pred'],
                                mode='lines',
                                name='Reta de Regressão Ajustada',
                                line=dict(color='#ef4444', width=3)
                            ))
                            st.plotly_chart(fig_biv, use_container_width=True)

                            direcao_biv = "aumenta" if rb['coef'] > 0 else "diminui"
                            st.info(
                                f"**Interpretação Executiva:** O modelo explica **{formatar_r2(rb['r2'])}** do comportamento de `{rb['col_y']}`. "
                                f"A cada **+1** unidade em `{rb['col_x']}`, `{rb['col_y']}` {direcao_biv} em média "
                                f"**{formatar_valor(abs(rb['coef']), prefixo=prefixo_rb)}** "
                                f"(equação: `{rb['col_y']} = {formatar_decimal(rb['intercept'], 4)} + ({formatar_decimal(rb['coef'], 4)}) × {rb['col_x']}`, "
                                f"n = {formatar_inteiro(rb['n_obs'])} pares válidos). "
                                + ("A relação é **estatisticamente significativa (p < 0,05)**." if rb.get('significativa')
                                   else "Atenção: a relação **não** é estatisticamente significativa (p ≥ 0,05).")
                            )

                # ── SUB-ABA 2: ESTATÍSTICAS AVANÇADAS & DIAGNÓSTICO ───────────
                with sub_estat:
                    st.markdown("#### 📊 Análise Estatística de Variabilidade, Quartis e Outliers")

                    col_estat_sel = st.selectbox("Selecione a Métrica Numérica para Diagnóstico:", col_num, key="sb_estat_metric")
                    is_moeda_estat = eh_coluna_monetaria(col_estat_sel)
                    prefixo_estat = "R$ " if is_moeda_estat else ""
                    st.caption(f"🏷️ Unidade identificada automaticamente: **{'Monetária (R$)' if is_moeda_estat else 'Quantidade / Numérica Geral'}**")

                    try:
                        estat_metric = calcular_estatisticas_resumo(df[col_estat_sel])

                        e1, e2, e3, e4 = st.columns(4)
                        cv_val = estat_metric['coef_variacao']
                        with e1:
                            if estat_metric['cv_valido']:
                                e1.metric(
                                    "📊 Coef. de Variação (CV)",
                                    formatar_percentual(cv_val, 1),
                                    help="Mede a dispersão relativa (Desvio Padrão ÷ Média). Até 30% indica dados homogêneos; acima de 50% indica alta volatilidade."
                                )
                                e1.caption(
                                    f"Dispersão alta (>50%) · Desv. Padrão: {formatar_valor(estat_metric['desvio_padrao'], prefixo=prefixo_estat)}" if cv_val > 50
                                    else f"Dispersão moderada (30%–50%) · Desv. Padrão: {formatar_valor(estat_metric['desvio_padrao'], prefixo=prefixo_estat)}" if cv_val > 30
                                    else f"Dispersão baixa (≤30%: dados homogêneos) · Desv. Padrão: {formatar_valor(estat_metric['desvio_padrao'], prefixo=prefixo_estat)}"
                                )
                            else:
                                e1.metric("📊 Desvio Padrão", formatar_valor(estat_metric['desvio_padrao'], prefixo=prefixo_estat))
                                e1.caption("CV não se aplica (métrica com valores ≤ 0)")

                        with e2:
                            col_outras = [c for c in col_num if c != col_estat_sel]
                            if col_outras:
                                col_corr_sel = st.selectbox("Comparar correlação com:", col_outras, key="sb_corr_metric")
                                corr_res = calcular_correlacao(df[col_estat_sel], df[col_corr_sel])
                                r_val = corr_res['r']
                                e2.metric(
                                    f"🔗 Correlação com '{col_corr_sel}'",
                                    f"{formatar_percentual(r_val * 100, 1, sinal=True)} ({formatar_decimal(r_val, 3)})" if not pd.isna(r_val) else "N/A",
                                    help="Correlação Linear de Pearson (r de -100% a +100%) e poder explicativo compartilhado (R²)."
                                )
                                r2_corr_txt = f"R² = {formatar_percentual((r_val ** 2) * 100, 1)} explicados · " if not pd.isna(r_val) else ""
                                e2.caption(f"{r2_corr_txt}Pearson {corr_res['interpretacao']} ({'p < 0,05 ✅' if corr_res['significativa'] else 'não signif. ⚠️'})")
                            else:
                                e2.metric("🔗 Correlação", "100,0% (1,000)")
                                e2.caption("Correlação de Pearson")

                        with e3:
                            # Critério estatístico real de outliers (Tukey/IQR), exibindo % e total de registros
                            e3.metric(
                                "⚠️ Outliers (Critério IQR)",
                                f"{formatar_percentual(estat_metric['outliers_iqr_pct'], 1)} ({formatar_inteiro(estat_metric['outliers_iqr_count'])})",
                                help="Porcentagem e quantidade de registros fora da faixa normal de Tukey (abaixo de Q1 − 1,5×IQR ou acima de Q3 + 1,5×IQR)."
                            )
                            e3.caption(
                                f"Faixa normal: {formatar_valor(estat_metric['limite_inferior'], prefixo=prefixo_estat)} a {formatar_valor(estat_metric['limite_superior'], prefixo=prefixo_estat)} "
                                f"· 95º perc.: {formatar_valor(estat_metric['p95'], prefixo=prefixo_estat)}"
                            )

                        with e4:
                            e4.metric(
                                "📈 Mediana (Valor Típico)",
                                formatar_valor(estat_metric['mediana'], prefixo=prefixo_estat),
                                help="Valor central (50% dos dados). Diferente da média, a mediana não é distorcida por valores extremos."
                            )
                            e4.caption(
                                f"Média: {formatar_valor(estat_metric['media'], prefixo=prefixo_estat)} · "
                                f"Mín–Máx: {formatar_valor(estat_metric['minimo'], prefixo=prefixo_estat)} a {formatar_valor(estat_metric['maximo'], prefixo=prefixo_estat)}"
                            )

                        st.markdown("---")

                        ocultar_extremos = st.checkbox(
                            "Ocultar no histograma a cauda extrema acima do 95º percentil (facilita visualizar distribuições muito assimétricas)",
                            value=False
                        )

                        ge1, ge2 = st.columns(2)

                        with ge1:
                            s_num_plot = pd.to_numeric(df[col_estat_sel], errors='coerce').dropna()
                            if ocultar_extremos:
                                s_num_plot = s_num_plot[s_num_plot <= estat_metric['p95']]
                            sufixo_t = " (até o 95º percentil)" if ocultar_extremos else " (base completa)"

                            fig_dist = go.Figure()
                            fig_dist.add_trace(go.Histogram(
                                x=s_num_plot,
                                nbinsx=40,
                                name="Frequência",
                                marker_color="#3b82f6",
                                opacity=0.75
                            ))
                            fig_dist.add_vline(
                                x=estat_metric['media'],
                                line_dash="dash",
                                line_color="#ef4444",
                                annotation_text=f"Média: {formatar_valor(estat_metric['media'], prefixo=prefixo_estat)}",
                                annotation_position="top left"
                            )
                            fig_dist.add_vline(
                                x=estat_metric['mediana'],
                                line_dash="dash",
                                line_color="#10b981",
                                annotation_text=f"Mediana: {formatar_valor(estat_metric['mediana'], prefixo=prefixo_estat)}",
                                annotation_position="bottom right"
                            )
                            fig_dist.update_layout(
                                title=f"Distribuição de '{col_estat_sel}'{sufixo_t}",
                                xaxis_title=col_estat_sel,
                                yaxis_title="Frequência",
                                template="plotly_dark",
                                height=420
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                        with ge2:
                            ordem_faixas = [
                                "Outlier Inferior", "Q1 (até 25%)", "Q2 (25%–50%)",
                                "Q3 (50%–75%)", "Q4 (75%–100%)", "Outlier Superior"
                            ]
                            classificacao_quartis = classificar_quartis_outliers(df[col_estat_sel])
                            cont_quartis = classificacao_quartis.value_counts().reindex(
                                [f for f in ordem_faixas if f in classificacao_quartis.values]
                            )

                            fig_quartis = px.pie(
                                values=cont_quartis.values,
                                names=cont_quartis.index,
                                category_orders={"names": ordem_faixas},
                                title="Segmentação por Quartis & Outliers (Critério IQR)",
                                template="plotly_dark",
                                hole=0.35,
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            fig_quartis.update_traces(textposition="inside", textinfo="percent+label", sort=False)
                            fig_quartis.update_layout(height=420)
                            st.plotly_chart(fig_quartis, use_container_width=True)

                        st.markdown("##### 📋 Tabela Resumo de Quartis, Assimetria & Limites de Outlier")
                        df_resumo_q = pd.DataFrame({
                            "Parâmetro Estatístico": [
                                "Registros Válidos / Ausentes",
                                "Mínimo / Máximo",
                                "Q1 (25% dos dados ≤ este valor)",
                                "Q2 (Mediana / 50%)",
                                "Q3 (75% dos dados ≤ este valor)",
                                "Percentil 95 (5% maiores estão acima)",
                                "IQR (Intervalo Interquartil = Q3 − Q1)",
                                "Limite Inferior para Outlier (Q1 − 1,5×IQR)",
                                "Limite Superior para Outlier (Q3 + 1,5×IQR)",
                                "Outliers IQR (Inferiores / Superiores / Total)",
                                "Assimetria (Skewness)"
                            ],
                            "Valor Calculado": [
                                f"{formatar_inteiro(estat_metric['contagem'])} válidos / {formatar_inteiro(estat_metric['ausentes'])} ausentes",
                                f"{formatar_valor(estat_metric['minimo'], prefixo=prefixo_estat)} a {formatar_valor(estat_metric['maximo'], prefixo=prefixo_estat)}",
                                formatar_valor(estat_metric['q1'], prefixo=prefixo_estat),
                                formatar_valor(estat_metric['q2'], prefixo=prefixo_estat),
                                formatar_valor(estat_metric['q3'], prefixo=prefixo_estat),
                                formatar_valor(estat_metric['p95'], prefixo=prefixo_estat),
                                formatar_valor(estat_metric['iqr'], prefixo=prefixo_estat),
                                formatar_valor(estat_metric['limite_inferior'], prefixo=prefixo_estat),
                                formatar_valor(estat_metric['limite_superior'], prefixo=prefixo_estat),
                                f"{formatar_inteiro(estat_metric['outliers_inferiores'])} inf. / {formatar_inteiro(estat_metric['outliers_superiores'])} sup. / {formatar_inteiro(estat_metric['outliers_iqr_count'])} total ({formatar_percentual(estat_metric['outliers_iqr_pct'], 2)})",
                                f"{formatar_decimal(estat_metric['assimetria'], 2)} ({estat_metric['assimetria_texto']})"
                            ]
                        })
                        st.dataframe(df_resumo_q, use_container_width=True, hide_index=True)

                        with st.expander("🔍 Parecer Estatístico & Recomendações Prescritivas", expanded=True):
                            comp_mm = (
                                "praticamente igual à mediana, indicando distribuição equilibrada"
                                if abs(estat_metric['assimetria']) < 0.5
                                else (
                                    "acima da mediana (puxada por valores altos)"
                                    if estat_metric['media'] > estat_metric['mediana']
                                    else "abaixo da mediana (puxada por valores baixos)"
                                )
                            )
                            st.markdown(f"""
                            - **Variabilidade dos Dados:** {f"Coeficiente de variação de **{formatar_percentual(cv_val, 2)}** indica dispersão relativa **{interpretar_cv(cv_val)}**." if estat_metric['cv_valido'] else f"Desvio padrão de **{formatar_valor(estat_metric['desvio_padrao'], prefixo=prefixo_estat)}** ao redor da média."}
                            - **Presença de Outliers (Tukey/IQR):** Identificados **{formatar_inteiro(estat_metric['outliers_iqr_count'])}** registros fora dos limites esperados ({formatar_percentual(estat_metric['outliers_iqr_pct'], 1)} do total: {formatar_inteiro(estat_metric['outliers_superiores'])} acima do limite superior e {formatar_inteiro(estat_metric['outliers_inferiores'])} abaixo do inferior).
                            - **Média vs Mediana e Assimetria:** A média ({formatar_valor(estat_metric['media'], prefixo=prefixo_estat)}) está {comp_mm} ({formatar_valor(estat_metric['mediana'], prefixo=prefixo_estat)}). A distribuição é **{estat_metric['assimetria_texto']}** (skewness = {formatar_decimal(estat_metric['assimetria'], 2)}).
                            """)

                    except Exception as e_estat:
                        st.error(f"Erro ao processar estatísticas de '{col_estat_sel}': {e_estat}")
