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

from data_science import (
    formatar_valor,
    eh_coluna_monetaria,
    obter_prefixo_coluna,
    calcular_estatisticas_resumo,
    classificar_quartis_outliers,
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
    substituicoes = {
        '•': '-', '–': '-', '—': '-', '“': '"', '”': '"', '‘': "'", '’': "'",
        '🚀': '', '🎯': '', '📌': '', '💡': '', '✨': '', '🧠': '', '🔬': '',
        '👀': '', '📊': '', '⚠️': 'ALERTA:', '📈': '', '📉': '', '🔑': '',
        '✅': '', '❌': '', '📋': '', '🏆': '', '💰': '', '📅': ''
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
    pdf.cell(65, 7, f'- Total de Registros: {total_linhas:,}', border=0)
    pdf.cell(55, 7, f'- Total de Colunas: {total_colunas}', border=0)
    pdf.cell(70, 7, f'- Nulos Tratados via ETL: {nulos_tratados:,}', border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, '2. PAINEL VISUAL DE INDICADORES CHAVE', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    tmp_path = None
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        df_top = df.groupby(var_cat)[var_num].mean().reset_index().sort_values(by=var_num, ascending=False).head(8)
        ax1.bar(df_top[var_cat].astype(str), df_top[var_num], color='#2563eb')
        ax1.set_title(f'Media de {var_num} por {var_cat}', fontsize=8, fontweight='bold')
        ax1.tick_params(axis='x', rotation=35, labelsize=6)
        ax1.tick_params(axis='y', labelsize=6)

        ax2.pie(df_top[var_num].clip(lower=0), labels=df_top[var_cat].astype(str), autopct='%1.1f%%', textprops={'fontsize': 6})
        ax2.set_title('Distribuicao Proporcional', fontsize=8, fontweight='bold')

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
# FUNÇÃO DE ETL AUTOMÁTICO (COM AUDITORIA DE NULOS REAIS)
# ---------------------------------------------------------
def executar_etl_automatico(df_raw):
    df_clean = df_raw.copy()
    duplicatas_removidas = int(df_clean.duplicated().sum())
    df_clean = df_clean.drop_duplicates()

    # Limpeza segura de espaços em textos SEM transformar NaN na string literal 'nan'
    colunas_texto = df_clean.select_dtypes(include=['object']).columns
    for col in colunas_texto:
        df_clean[col] = df_clean[col].where(
            df_clean[col].isna(),
            df_clean[col].astype(str).str.strip()
        )
        df_clean[col] = df_clean[col].replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'null': np.nan})

    # Registra a contagem real de valores ausentes antes da imputação automática
    nulos_originais = df_clean.isnull().sum().to_dict()

    # Identificação inteligente de datas
    for col in df_clean.columns:
        col_lower = col.lower()
        eh_duracao_ou_numero = any(p in col_lower for p in [
            'tempo_', '_horas', 'horas_', '_minutos', 'minutos_', 'hours', 'minutes', 'age', 'idade'
        ])
        if not eh_duracao_ou_numero and any(k in col_lower for k in ('date', 'dt', 'data_registro', 'timestamp', 'data')):
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            except Exception:
                pass

    # Garante conversão de colunas numéricas conhecidas
    for col in df_clean.columns:
        if any(p in col.lower() for p in ['tempo_', 'horas_', 'uso_ia', 'pontuacao_', 'indice_', 'idade', 'score', 'ability', 'skills']):
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    if 'data_hora_real' in df_clean.columns and 'data_hora_agendada' in df_clean.columns:
        if (pd.api.types.is_datetime64_any_dtype(df_clean['data_hora_real']) and
                pd.api.types.is_datetime64_any_dtype(df_clean['data_hora_agendada'])):
            df_clean['atraso_real_minutos'] = (
                (df_clean['data_hora_real'] - df_clean['data_hora_agendada'])
                .dt.total_seconds() / 60.0
            ).round(1)

    if 'flag_pontualidade' in df_clean.columns:
        df_clean['pontualidade_pct'] = (
            df_clean['flag_pontualidade'].astype(str).str.lower()
            .map({'true': 100, 'false': 0, '1': 100, '0': 0})
            .fillna(0)
        )

    # Tratamento de valores ausentes (Imputação)
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].ffill().bfill()
            else:
                df_clean[col] = df_clean[col].fillna("Não informado")

    # Extrai Ano e Mês de colunas datetime
    colunas_datas = df_clean.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns
    for col in colunas_datas:
        prefixo = col.replace('_datetime', '').replace('_date', '').replace('_data', '')
        df_clean[f'{prefixo}_ano'] = df_clean[col].dt.year
        df_clean[f'{prefixo}_mes'] = df_clean[col].dt.month_name().map(MESES_PT).fillna(df_clean[col].dt.month_name())
        nulos_originais.setdefault(f'{prefixo}_ano', 0)
        nulos_originais.setdefault(f'{prefixo}_mes', 0)

    df_clean.attrs['nulos_originais'] = nulos_originais
    df_clean.attrs['duplicatas_removidas'] = duplicatas_removidas
    return df_clean

# ---------------------------------------------------------
# FUNÇÃO DE CARREGAMENTO SQL
# ---------------------------------------------------------
def carregar_dados_sql(db_type, host, port, user, password, database, tabela):
    try:
        if db_type == "PostgreSQL":
            url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        elif db_type == "MySQL":
            url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
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
# INTEGRAÇÃO COM IA MULTIMOTOR
# ---------------------------------------------------------
def construir_prompt_insights(df_info_str, df_describe_str, df_head_str, tipo_analise="Automático (Geral)", pergunta_usuario=""):
    info_safe = df_info_str[:1500]
    desc_safe = df_describe_str[:2500]
    head_safe = df_head_str[:1200]

    eh_sob_demanda = bool(pergunta_usuario and pergunta_usuario.strip()) or (tipo_analise and tipo_analise != "Automático (Geral)")

    if not eh_sob_demanda:
        prompt = (
            "Atue como um Especialista em Analytics e Designer de Relatórios Executivos.\n"
            "Analise a estrutura de dados fornecida abaixo e gere um parecer estruturado "
            "estritamente em Português do Brasil de alto nível executivo e estratégico.\n\n"
            "Siga exatamente a seguinte estrutura e estilo em Markdown:\n\n"
            "# InnovaNexus - Relatório Executivo de Dados\n"
            "*Parecer Prescritivo de Negócio, Diagnóstico de ETL e Visualização de Indicadores*\n\n"
            "---\n\n"
            "> **🎯 Objetivo Estratégico:** Identificar padrões críticos, correlações e oportunidades de melhoria "
            "a partir dos indicadores reais da base de dados.\n\n"
            "---\n\n"
            "## 📊 1. Indicadores-Chave de Desempenho (KPIs)\n\n"
            "Apresente uma tabela markdown com os KPIs extraídos ou inferidos da base de dados.\n\n"
            "---\n\n"
            "## 💡 2. Insights Estratégicos e Diagnóstico\n\n"
            "Destaque em subseções numeradas os 3 diagnósticos principais encontrados.\n\n"
            "---\n\n"
            "## 🚀 3. Ações Prescritivas e Recomendações\n\n"
            "Forneça 3 recomendações acionáveis para tomada de decisão.\n\n"
            "--- DADOS PARA ANÁLISE ---\n"
            f"Tipos de Colunas:\n{info_safe}\n\n"
            f"Resumo Estatístico:\n{desc_safe}\n\n"
            f"Amostra de Dados:\n{head_safe}\n"
        )
    else:
        foco_tipo = f"- **Foco da Análise Solicitada:** {tipo_analise}\n" if tipo_analise and tipo_analise != "Automático (Geral)" else ""
        foco_pergunta = f"- **Pergunta / Insight Específico Desejado:** {pergunta_usuario.strip()}\n" if pergunta_usuario and pergunta_usuario.strip() else ""

        prompt = (
            "Atue como um Especialista em Ciência de Dados, Analytics e Inteligência de Negócios.\n"
            "O usuário solicitou uma ANÁLISE SOB DEMANDA específica sobre o dataset enviado.\n"
            "Sua resposta deve ser gerada estritamente a partir dos dados fornecidos e refletir com precisão o que foi solicitado.\n\n"
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
            "--- DADOS PARA ANÁLISE ---\n"
            f"Tipos de Colunas:\n{info_safe}\n\n"
            f"Resumo Estatístico:\n{desc_safe}\n\n"
            f"Amostra de Dados:\n{head_safe}\n"
        )
    return prompt

def gerar_insights_ia(motor, api_key, df_info_str, df_describe_str, df_head_str, tipo_analise="Automático (Geral)", pergunta_usuario=""):
    try:
        prompt = construir_prompt_insights(df_info_str, df_describe_str, df_head_str, tipo_analise, pergunta_usuario)
        system_msg = "Você é um analista de dados sênior especialista em business intelligence e ciência de dados."

        # ── 1. MOTOR GROQ ─────────────────────────────────────────────────────
        if motor == "Groq (Rápido/Gratuito)":
            key_groq = api_key.strip()
            if not key_groq:
                return None, (
                    "🔑 Chave Groq não configurada.\n\n"
                    "Obtenha uma chave gratuita em https://console.groq.com/keys "
                    "e insira-a no campo 'Chave de API' no painel lateral."
                )

            client = OpenAI(
                api_key=key_groq,
                base_url="https://api.groq.com/openai/v1"
            )

            modelos_groq = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ]

            ultimo_erro = None
            for modelo in modelos_groq:
                try:
                    response = client.chat.completions.create(
                        model=modelo,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user",   "content": prompt}
                        ],
                        max_tokens=2048,
                        temperature=0.7,
                        stream=False
                    )
                    if response and response.choices and len(response.choices) > 0:
                        return response.choices[0].message.content, None
                except Exception as e:
                    err_str = str(e)
                    ultimo_erro = err_str
                    if "401" in err_str or "authentication" in err_str.lower() or "invalid_api_key" in err_str.lower():
                        return None, (
                            "❌ Chave Groq inválida ou expirada (erro 401).\n\n"
                            "Obtenha uma nova chave em:\n"
                            "➡️ https://console.groq.com/keys\n\n"
                            "A chave começa com **gsk_** — cole-a no campo 'Chave de API' na sidebar."
                        )
                    continue

            return None, (
                f"❌ Erro na API da Groq: {ultimo_erro}\n\n"
                "Verifique sua chave em https://console.groq.com/keys e confirme se ela possui cota disponível."
            )

        # ── 2. MOTOR GEMINI ───────────────────────────────────────────────────
        elif motor == "Gemini (Google)":
            key = api_key.strip()
            if not key:
                return None, (
                    "🔑 Chave Gemini não configurada.\n\n"
                    "Obtenha uma chave no Google AI Studio em https://aistudio.google.com/apikey"
                )

            # Tenta via SDK oficial google-genai (models.generate_content)
            try:
                from google import genai as ggenai
                gc = ggenai.Client(api_key=key)
                for modelo_sdk in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                    try:
                        resp_sdk = gc.models.generate_content(
                            model=modelo_sdk,
                            contents=f"{system_msg}\n\n{prompt}"
                        )
                        if resp_sdk and getattr(resp_sdk, "text", None):
                            return resp_sdk.text, None
                    except Exception as e_sdk:
                        if any(t in str(e_sdk).lower() for t in ["not found", "404", "deprecated"]):
                            continue
                        break
            except Exception:
                pass

            # Fallback REST API direta
            modelos_rest = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            headers = {'Content-Type': 'application/json'}
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"{system_msg}\n\n{prompt}"}]}
                ],
                "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.7}
            }

            for modelo_rest in modelos_rest:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{modelo_rest}:generateContent?key={key}"
                )
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=60)
                    if resp.status_code == 200:
                        dados = resp.json()
                        candidatos = dados.get("candidates", [])
                        if candidatos:
                            partes = candidatos[0].get("content", {}).get("parts", [])
                            if partes:
                                return partes[0].get("text", ""), None
                        return None, "Resposta vazia do Gemini. Tente novamente."
                    elif resp.status_code in (404, 400):
                        continue
                    elif resp.status_code == 403:
                        return None, "❌ Acesso negado (HTTP 403). Verifique se a chave do Google AI Studio é válida."
                    elif resp.status_code == 429:
                        return None, "⏳ Limite de requisições atingido (quota). Aguarde 1 minuto e tente novamente."
                    else:
                        try:
                            msg = resp.json().get('error', {}).get('message', resp.text[:300])
                        except Exception:
                            msg = resp.text[:300]
                        return None, f"Erro HTTP {resp.status_code} da API Gemini: {msg}"
                except Exception as e_rest:
                    return None, f"Erro ao chamar o Gemini: {str(e_rest)}"

            return None, "Nenhum modelo Gemini respondeu. Confirme que sua chave está ativa em aistudio.google.com."

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
        ["Groq (Rápido/Gratuito)", "Gemini (Google)"]
    )

    chave_personalizada = st.sidebar.text_input(
        "Chave de API (Opcional)",
        type="password",
        help="Deixe em branco para usar a chave configurada no sistema para o motor selecionado."
    )

    chave_ativa = chave_personalizada.strip() if chave_personalizada.strip() else CHAVES_PADRAO.get(motor_selecionado, "")
    chave_valida = bool(chave_ativa)

    if motor_selecionado == "Gemini (Google)":
        if chave_valida:
            st.sidebar.success("✅ API Gemini: Pronta")
        else:
            st.sidebar.warning("⚠️ Chave Gemini não configurada. Insira sua chave do Google AI Studio.")
    else:
        if chave_valida:
            st.sidebar.success(f"✅ API {motor_selecionado.split()[0]}: Pronta")
        else:
            st.sidebar.warning(f"⚠️ Chave {motor_selecionado.split()[0]} não configurada.")

    links_chave = {
        "Groq (Rápido/Gratuito)": "https://console.groq.com/keys",
        "Gemini (Google)":        "https://aistudio.google.com/apikey",
    }
    st.sidebar.caption(f"🔑 [Obter chave {motor_selecionado.split()[0]}]({links_chave[motor_selecionado]})")

    # ── FONTE DE DADOS ─────────────────────────────────────────────────────────
    st.markdown("---")
    fonte_dados = st.radio(
        "Escolha a fonte de entrada de dados:",
        ["Upload de Arquivo (.csv)", "Conexão com Banco de Dados (SQL)"],
        horizontal=True
    )

    df_base = None

    if fonte_dados == "Upload de Arquivo (.csv)":
        arquivo_carregado = st.file_uploader(
            "Selecione um ficheiro CSV para análise",
            type=["csv"],
            key="uploader_csv"
        )
        if arquivo_carregado is not None:
            try:
                df_raw = pd.read_csv(arquivo_carregado)
                if df_raw.shape[1] == 1 and ',' in str(df_raw.columns[0]):
                    arquivo_carregado.seek(0)
                    linhas = [line.decode('utf-8', errors='replace').strip() for line in arquivo_carregado.readlines()]
                    linhas_limpas = [l[1:-1] if l.startswith('"') and l.endswith('"') else l for l in linhas]
                    df_raw = pd.read_csv(StringIO('\n'.join(linhas_limpas)))
                df_base = executar_etl_automatico(df_raw)
                st.success(
                    f"✅ Ficheiro carregado e processado via ETL! "
                    f"**{df_base.shape[0]:,} registros** × **{df_base.shape[1]} colunas** identificadas."
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
                    df_sql, erro = carregar_dados_sql(db_type, db_host, db_port, db_user, db_pass, db_name, db_table)
                    if erro:
                        st.error(f"Erro na conexão: {erro}")
                    else:
                        df_sql = executar_etl_automatico(df_sql)
                        st.success("✅ Dados carregados via ETL a partir da base de dados!")
                        st.session_state['df_sql'] = df_sql
            else:
                st.warning("Por favor, informe o nome da tabela.")

        if 'df_sql' in st.session_state:
            df_base = st.session_state['df_sql']

    # ── FILTRO DINÂMICO DE SEGMENTAÇÃO NA SIDEBAR ──────────────────────────────
    df = df_base
    if df_base is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Filtro de Segmentação")
        col_cat_filtro = [
            c for c in df_base.select_dtypes(include=['object', 'category']).columns
            if 'id' not in c.lower() and df_base[c].nunique() <= 30
        ]
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
                    df.attrs = df_base.attrs
                    st.sidebar.caption(f"🔎 Exibindo **{len(df):,}** de **{len(df_base):,}** registros.")

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Sair (Logout)"):
        realizar_logout()

    # ══════════════════════════════════════════════════════════════════════════
    # DASHBOARDS — só renderiza se há dados carregados
    # ══════════════════════════════════════════════════════════════════════════
    if df is not None and not df.empty:
        st.markdown("---")

        # Identificação inteligente de colunas numéricas (exclui IDs puros do topo)
        todas_num = df.select_dtypes(include=['number']).columns.tolist()
        col_num = [
            c for c in todas_num
            if c.lower() not in ('row_id', 'id', 'index', 'codigo', 'código')
            and not c.lower().endswith('_ano')
        ]
        if not col_num:
            col_num = todas_num

        col_cat = [
            c for c in df.select_dtypes(include=['object', 'category']).columns.tolist()
            if 'id' not in c.lower()
        ]
        if not col_cat:
            col_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()

        col_data = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()

        # Auditoria de nulos originais tratados pelo ETL
        nulos_orig_map = df.attrs.get('nulos_originais', {})
        nulos_orig_series = pd.Series([nulos_orig_map.get(c, 0) for c in df.columns], index=df.columns)
        total_nulos_tratados = int(nulos_orig_series.sum())

        info_df = pd.DataFrame({
            "Coluna":                  df.columns,
            "Tipo de Dado":            [str(d) for d in df.dtypes],
            "Nulos Brutos (Tratados)": nulos_orig_series.values,
            "% Nulos Brutos":          (nulos_orig_series.values / max(1, len(df_base)) * 100).round(1),
            "Valores Únicos":          df.nunique().values,
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

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("📦 Registros Ativos",     f"{df.shape[0]:,}")
            k2.metric("📐 Colunas",              f"{df.shape[1]}")
            k3.metric("🔢 Métricas Numéricas",   f"{len(col_num)}")
            k4.metric("🏷️ Colunas Categóricas", f"{len(col_cat)}")
            k5.metric("🛠️ Nulos Tratados (ETL)", f"{total_nulos_tratados:,}")

            st.markdown("---")

            n_linhas = st.slider(
                "Número de linhas a exibir:",
                min_value=10,
                max_value=min(500, max(10, df.shape[0])),
                value=min(50, df.shape[0]),
                step=10
            )
            st.dataframe(df.head(n_linhas), use_container_width=True, height=500)

            csv_tratado = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Baixar Dataset Tratado (CSV)",
                data=csv_tratado,
                file_name="dataset_tratado.csv",
                mime="text/csv"
            )

        # ── TAB 2: DIAGNÓSTICO TÉCNICO ─────────────────────────────────────────
        with tab_diagnostico:
            st.subheader("🔬 Diagnóstico Técnico de Qualidade & Auditoria de ETL")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 📋 Estrutura e Qualidade (Auditoria Pré/Pós-ETL)")
                st.dataframe(info_df, use_container_width=True, height=400)

            with c2:
                if col_num:
                    st.markdown("##### 📈 Estatísticas Descritivas")
                    st.dataframe(df[col_num].describe().T.round(4), use_container_width=True, height=400)

            if total_nulos_tratados > 0:
                st.markdown("##### 🛠️ Valores Ausentes Identificados e Imputados pelo ETL por Coluna")
                ausentes = pd.DataFrame({
                    'Coluna': nulos_orig_series.index,
                    'Nulos Tratados': nulos_orig_series.values
                })
                ausentes = ausentes[ausentes['Nulos Tratados'] > 0].sort_values('Nulos Tratados', ascending=False)
                fig_missing = px.bar(
                    ausentes, x='Coluna', y='Nulos Tratados',
                    title="Colunas com Valores Ausentes Tratados Automaticamente no ETL",
                    template="plotly_dark",
                    color='Nulos Tratados',
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
                    color_continuous_scale="RdBu",
                    title="Correlação de Pearson",
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

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

            if not chave_valida:
                st.error(
                    f"⚠️ Configure uma Chave de API válida para o motor **{motor_selecionado}** "
                    f"no painel lateral esquerdo."
                )
            else:
                btn_texto = "✨ Gerar Parecer Executivo Automático" if modo_analise == "Automático (Padrão)" else "🎯 Gerar Análise Sob Demanda"
                if st.button(btn_texto, type="primary"):
                    with st.spinner(f"A processar dados via {motor_selecionado}... Aguarde."):
                        info_str     = info_df.to_string(index=False)
                        describe_str = df[col_num].describe().T.to_string() if col_num else "Nenhuma variável numérica."
                        head_str     = df.head(5).to_string(index=False)

                        insights, erro_ia = gerar_insights_ia(
                            motor_selecionado, chave_ativa, info_str, describe_str, head_str,
                            tipo_analise=tipo_analise_input,
                            pergunta_usuario=pergunta_usuario_input
                        )

                        if erro_ia:
                            st.error(f"❌ Falha ao gerar parecer:\n\n{erro_ia}")
                        else:
                            st.session_state['insights_ia'] = insights

                if 'insights_ia' in st.session_state:
                    st.markdown(st.session_state['insights_ia'])

                    if col_num and col_cat:
                        try:
                            var_num_pdf = st.session_state.get('dash_var_num', col_num[0])
                            var_cat_pdf = st.session_state.get('dash_var_cat', col_cat[0])
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
                col_sel1, col_sel2, col_sel3 = st.columns(3)
                var_num  = col_sel1.selectbox("📈 Métrica principal (Eixo Y):", col_num, key="dash_var_num")
                var_cat  = col_sel2.selectbox("🏷️ Categoria (Eixo X):", col_cat, key="dash_var_cat")
                var_num2 = col_sel3.selectbox(
                    "📉 2ª Métrica (comparativo):",
                    [c for c in col_num if c != var_num] or col_num
                )

                top_n = st.slider("Top N categorias:", min_value=5, max_value=30, value=10, step=5)

                df_top = (
                    df.groupby(var_cat)[var_num]
                    .mean()
                    .reset_index()
                    .sort_values(by=var_num, ascending=False)
                    .head(top_n)
                )

                st.markdown("---")

                # Linha 1: Barras + Pizza
                r1c1, r1c2 = st.columns([3, 2])
                with r1c1:
                    fig_bar = px.bar(
                        df_top, x=var_cat, y=var_num,
                        title=f"🏆 Top {top_n}: Média de '{var_num}' por '{var_cat}'",
                        template="plotly_dark",
                        color=var_num,
                        color_continuous_scale="Blues",
                        text_auto='.2s'
                    )
                    fig_bar.update_layout(xaxis_tickangle=-35, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

                with r1c2:
                    fig_pie = px.pie(
                        df_top, values=var_num, names=var_cat,
                        title=f"🥧 Distribuição Proporcional — '{var_num}'",
                        template="plotly_dark",
                        hole=0.35
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Linha 2: Box Plot + Histograma
                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    cats_top = df_top[var_cat].tolist()
                    df_box = df[df[var_cat].isin(cats_top)]
                    fig_box = px.box(
                        df_box, x=var_cat, y=var_num,
                        title=f"📦 Box Plot: '{var_num}' por '{var_cat}'",
                        template="plotly_dark",
                        color=var_cat,
                        notched=False
                    )
                    fig_box.update_layout(xaxis_tickangle=-35, showlegend=False)
                    st.plotly_chart(fig_box, use_container_width=True)

                with r2c2:
                    fig_hist = px.histogram(
                        df, x=var_num,
                        title=f"📊 Distribuição: '{var_num}'",
                        template="plotly_dark",
                        nbins=40,
                        color_discrete_sequence=["#3b82f6"],
                        marginal="rug"
                    )
                    fig_hist.update_layout(bargap=0.05)
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Linha 3: Scatter + Barras Horizontais
                r3c1, r3c2 = st.columns(2)
                with r3c1:
                    if len(col_num) >= 2:
                        try:
                            import statsmodels  # noqa: F401
                            _trendline = "ols"
                        except ImportError:
                            _trendline = None

                        df_sc_sample = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df
                        fig_scatter = px.scatter(
                            df_sc_sample, x=var_num, y=var_num2,
                            color=var_cat if df[var_cat].nunique() <= 20 else None,
                            title=f"🔵 Dispersão: '{var_num}' vs '{var_num2}'",
                            template="plotly_dark",
                            trendline=_trendline,
                            opacity=0.7
                        )
                        fig_scatter.update_layout(showlegend=False)
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("Scatter plot requer ao menos 2 colunas numéricas.")

                with r3c2:
                    if len(col_num) >= 2 and var_num2 in df.columns:
                        df_comp = df.groupby(var_cat)[[var_num, var_num2]].mean().reset_index()
                        df_comp = df_comp.sort_values(var_num, ascending=False).head(top_n)
                        df_melt = df_comp.melt(
                            id_vars=var_cat, value_vars=[var_num, var_num2],
                            var_name="Métrica", value_name="Valor"
                        )
                        fig_comp = px.bar(
                            df_melt, y=var_cat, x="Valor", color="Métrica",
                            orientation='h',
                            title=f"📊 Comparativo: '{var_num}' vs '{var_num2}'",
                            template="plotly_dark",
                            barmode="group"
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                    else:
                        fig_hbar = px.bar(
                            df_top.sort_values(var_num),
                            y=var_cat, x=var_num,
                            orientation='h',
                            title=f"📊 Ranking Horizontal: '{var_num}'",
                            template="plotly_dark",
                            color=var_num,
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig_hbar, use_container_width=True)

                # Linha 4: Série Temporal + Treemap
                r4c1, r4c2 = st.columns(2)
                with r4c1:
                    if col_data:
                        col_dt = col_data[0]
                        df_ts = df.groupby(col_dt)[var_num].mean().reset_index().sort_values(col_dt)
                        fig_ts = px.line(
                            df_ts, x=col_dt, y=var_num,
                            title=f"📅 Evolução Média: '{var_num}' ao longo do tempo",
                            template="plotly_dark",
                            markers=True
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)
                    else:
                        cats_top2 = df_top[var_cat].tolist()
                        df_vio = df[df[var_cat].isin(cats_top2)]
                        fig_vio = px.violin(
                            df_vio, x=var_cat, y=var_num,
                            box=True, points="outliers",
                            title=f"🎻 Violin Plot: '{var_num}' por '{var_cat}'",
                            template="plotly_dark",
                            color=var_cat
                        )
                        fig_vio.update_layout(showlegend=False, xaxis_tickangle=-35)
                        st.plotly_chart(fig_vio, use_container_width=True)

                with r4c2:
                    if len(col_num) >= 1 and len(col_cat) >= 1:
                        df_tree = (
                            df.groupby(var_cat)[var_num]
                            .mean()
                            .reset_index()
                            .sort_values(var_num, ascending=False)
                            .head(min(20, df[var_cat].nunique()))
                        )
                        fig_tree = px.treemap(
                            df_tree,
                            path=[var_cat],
                            values=var_num,
                            title=f"🌳 Treemap: Proporção Média de '{var_num}' por '{var_cat}'",
                            template="plotly_dark",
                            color=var_num,
                            color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig_tree, use_container_width=True)

                st.markdown("---")
                st.markdown("##### 📋 Tabela de Resumo Estatístico por Categoria")
                df_resumo = df.groupby(var_cat)[var_num].agg(
                    Contagem='count',
                    Média='mean',
                    Mediana='median',
                    Mínimo='min',
                    Máximo='max',
                    DesvPad='std'
                ).round(2).reset_index().sort_values('Média', ascending=False)
                st.dataframe(df_resumo, use_container_width=True, height=350)

        # ── TAB 5: CIÊNCIA DE DADOS & REGRESSÃO ───────────────────────────────
        with tab_ciencia_dados:
            st.subheader("📈 Ciência de Dados: Regressão Linear & Estatística Avançada")
            st.markdown(
                "Módulo preditivo e estatístico integrado ao InnovaNexus. "
                "Execute modelagem de **Regressão Linear** para projeções futuras temporais ou análise de correlação bivariada, "
                "além de diagnósticos detalhados de **Quartis, Outliers e Variabilidade**."
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
                            if c not in col_data and any(k in c.lower() for k in ('date', 'data', 'tempo', 'dt', 'periodo', 'ano', 'mes', 'year'))
                        ]

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
                            opcoes_freq = ["Anual", "Mensal", "Trimestral", "Diário"] if eh_ano_inteiro else ["Mensal", "Anual", "Trimestral", "Diário"]
                            freq_sel = st.selectbox("⏱️ Periodicidade:", opcoes_freq)
                            freq_map = {"Mensal": "M", "Diário": "D", "Trimestral": "Q", "Anual": "Y"}
                        with c_reg4:
                            default_agg_idx = 0 if is_moeda_sel else 1
                            agg_sel = st.selectbox("∑ Agregação:", ["Soma", "Média"], index=default_agg_idx)
                            agg_val = "sum" if agg_sel == "Soma" else "mean"

                        st.caption(f"🏷️ Unidade identificada automaticamente: **{'Monetária (R$)' if is_moeda_sel else 'Quantidade / Numérica Geral'}**")

                        c_p1, c_p2 = st.columns([2, 1])
                        with c_p1:
                            n_periodos_sel = st.slider(
                                "Quantos períodos futuros deseja prever?",
                                min_value=1,
                                max_value=24,
                                value=6,
                                help="Número de passos futuros projetados com base na tendência linear dos dados históricos."
                            )
                        with c_p2:
                            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                            btn_gerar_prev = st.button("🔮 Gerar Previsão Temporal", type="primary")

                        if btn_gerar_prev:
                            try:
                                with st.spinner("Treinando modelo de regressão linear e calculando projeções..."):
                                    res_prev = executar_previsao_temporal(
                                        df, col_data_sel, col_valor_sel,
                                        n_periodos=n_periodos_sel,
                                        freq=freq_map[freq_sel],
                                        agregacao=agg_val
                                    )
                                    st.session_state['res_prev_temp'] = res_prev
                            except Exception as e:
                                st.error(f"❌ Não foi possível executar a regressão temporal: {e}")

                        if 'res_prev_temp' in st.session_state:
                            rp = st.session_state['res_prev_temp']
                            df_hist = rp['df_historico']
                            df_prev = rp['df_previsoes']
                            is_moeda_res = eh_coluna_monetaria(rp['col_valor'])
                            prefixo_res = "R$ " if is_moeda_res else ""
                            fmt_data = rp.get('formato_data', '%b/%Y')

                            st.markdown("---")
                            st.markdown("#### 📈 Qualidade do Ajuste do Modelo (Regressão Linear)")

                            m1, m2, m3, m4 = st.columns(4)
                            with m1:
                                m1.metric("R² (Determinação)", f"{rp['r2']:.3f}")
                                m1.caption("Quanto mais próximo de 1, melhor o ajuste da tendência")
                            with m2:
                                m2.metric("MAE (Erro Médio Absoluto)", formatar_valor(rp['mae'], prefixo=prefixo_res))
                                m2.caption("Erro médio absoluto nas unidades da métrica")
                            with m3:
                                m3.metric("RMSE (Raiz do Erro Médio²)", formatar_valor(rp['rmse'], prefixo=prefixo_res))
                                m3.caption("Penaliza desvios maiores")
                            with m4:
                                m4.metric("Coeficiente Angular", f"{formatar_valor(rp['coef'], prefixo=prefixo_res)} / período")
                                m4.caption(f"Variação média esperada por período ({freq_sel.lower()})")

                            st.markdown("#### 📊 Histórico e Previsão Projetada")
                            fig_prev = go.Figure()

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
                                title=f"Análise de Tendência e Previsão de '{rp['col_valor']}' ao Longo do Tempo",
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
                                df_prev_view['periodo_fmt'] = df_prev_view['data'].dt.strftime(fmt_data)
                                df_prev_view['valor_fmt'] = df_prev_view['valor_previsto'].apply(
                                    lambda v: formatar_valor(v, prefixo=prefixo_res)
                                )
                                df_prev_view['crescimento_fmt'] = df_prev_view['crescimento_pct'].apply(
                                    lambda x: f"{x:+.2f}%" if not pd.isna(x) else "-"
                                )
                                st.dataframe(
                                    df_prev_view[['periodo_fmt', 'valor_fmt', 'crescimento_fmt']].rename(columns={
                                        'periodo_fmt': 'Período',
                                        'valor_fmt': col_prev_label,
                                        'crescimento_fmt': 'Variação (%)'
                                    }),
                                    use_container_width=True,
                                    height=320
                                )

                            with c_bar:
                                st.markdown("##### 📊 Comparativo: Recente vs Previsão")
                                n_hist_comp = min(6, len(df_hist))
                                df_hist_comp = df_hist.tail(n_hist_comp)

                                fig_comp = go.Figure()
                                fig_comp.add_trace(go.Bar(
                                    x=df_hist_comp['data'].dt.strftime(fmt_data),
                                    y=df_hist_comp[rp['col_valor']],
                                    name='Histórico Recente',
                                    marker_color='#3b82f6'
                                ))
                                fig_comp.add_trace(go.Bar(
                                    x=df_prev['data'].dt.strftime(fmt_data),
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
                            r1, r2, r3, r4 = st.columns(4)
                            r1.metric("Soma / Acumulado Previsto", formatar_valor(rp['total_previsto'], prefixo=prefixo_res))
                            r2.metric("Média Periódica Prevista", formatar_valor(rp['media_mensal'], prefixo=prefixo_res))
                            r3.metric("Variação Esperada (Fim vs Histórico)", f"{rp['crescimento_total']:+.2f}%")
                            r4.metric("Último Ponto Histórico", formatar_valor(rp['ultimo_historico'], prefixo=prefixo_res))

                            with st.expander("ℹ️ Detalhes Matemáticos e Interpretação do Modelo"):
                                st.markdown(f"""
                                - **Equação da Reta:** $Y = {rp['intercept']:,.2f} + ({rp['coef']:,.2f}) \\times X$
                                - **Coeficiente Angular:** {formatar_valor(rp['coef'], prefixo=prefixo_res)} por período (variação periódica esperada)
                                - **Coeficiente de Determinação ($R^2$):** {rp['r2']:.3f} ({rp['r2'] * 100:.1f}% da variação explicada pela tendência linear)
                                - **Avaliação do Ajuste:** {'✅ Excelente ajuste linear (R² ≥ 0.70)' if rp['r2'] >= 0.7 else 'ℹ️ Ajuste moderado (0.50 ≤ R² < 0.70). Recomenda-se cautela em projeções de longo prazo.' if rp['r2'] >= 0.5 else '⚠️ Ajuste fraco (R² < 0.50). Dica: Utilize o Filtro de Segmentação na barra lateral para isolar um cenário específico.'}
                                """)

                    else:
                        st.markdown("##### 🔬 Regressão Linear entre Duas Variáveis Numéricas")
                        col_bx, col_by = st.columns(2)
                        with col_bx:
                            x_col_sel = st.selectbox("Variável Independente (X):", col_num, index=0)
                        with col_by:
                            y_opts = [c for c in col_num if c != x_col_sel] or col_num
                            y_col_sel = st.selectbox("Variável Dependente (Y):", y_opts, index=0)

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

                            st.markdown("---")
                            b1, b2, b3, b4 = st.columns(4)
                            b1.metric("R² (Determinação)", f"{rb['r2']:.3f}")
                            b2.metric("MAE", formatar_valor(rb['mae'], prefixo=prefixo_rb))
                            b3.metric("RMSE", formatar_valor(rb['rmse'], prefixo=prefixo_rb))
                            b4.metric("Coeficiente Angular", f"{rb['coef']:,.4f}")

                            df_biv_plot = df_biv.sample(min(3000, len(df_biv)), random_state=42) if len(df_biv) > 3000 else df_biv
                            fig_biv = px.scatter(
                                df_biv_plot, x=rb['col_x'], y=rb['col_y'],
                                title=f"Dispersão com Linha de Regressão: {rb['col_y']} vs {rb['col_x']}",
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

                            st.caption(f"Equação: **{rb['col_y']} = {rb['intercept']:,.4f} + ({rb['coef']:,.4f}) × {rb['col_x']}** (Amostra total: {rb['n_obs']:,} pares válidos)")

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
                            e1.metric("📊 Coef. de Variação", f"{cv_val:.2f}%")
                            e1.caption("Alta (>50%)" if cv_val > 50 else "Moderada (>30%)" if cv_val > 30 else "Baixa (≤30%)")

                        with e2:
                            col_outras = [c for c in col_num if c != col_estat_sel]
                            if col_outras:
                                col_corr_sel = st.selectbox("Comparar correlação com:", col_outras, key="sb_corr_metric")
                                s1 = pd.to_numeric(df[col_estat_sel], errors='coerce')
                                s2 = pd.to_numeric(df[col_corr_sel], errors='coerce')
                                corr_par = s1.corr(s2)
                                e2.metric(f"🔗 Correlação com '{col_corr_sel}'", f"{corr_par:.3f}" if not pd.isna(corr_par) else "N/A")
                            else:
                                e2.metric("🔗 Correlação", "1.000")
                            e2.caption("Correlação de Pearson")

                        with e3:
                            e3.metric("⚠️ Outliers (>95%)", f"{estat_metric['outliers_p95_pct']:.1f}%")
                            e3.caption(f"{estat_metric['outliers_p95_count']:,} registros no 95º percentil")

                        with e4:
                            e4.metric("📈 Mediana", formatar_valor(estat_metric['mediana'], prefixo=prefixo_estat))
                            e4.caption(f"Média: {formatar_valor(estat_metric['media'], prefixo=prefixo_estat)}")

                        st.markdown("---")

                        ge1, ge2 = st.columns(2)

                        with ge1:
                            limite_p95 = estat_metric['p95']
                            s_filtrada = df[pd.to_numeric(df[col_estat_sel], errors='coerce') <= limite_p95][col_estat_sel]

                            fig_dist = go.Figure()
                            fig_dist.add_trace(go.Histogram(
                                x=s_filtrada,
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
                                title=f"Distribuição de '{col_estat_sel}' (até o 95º percentil)",
                                xaxis_title=col_estat_sel,
                                yaxis_title="Frequência",
                                template="plotly_dark",
                                height=420
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                        with ge2:
                            classificacao_quartis = classificar_quartis_outliers(df[col_estat_sel])
                            cont_quartis = classificacao_quartis.value_counts()

                            fig_quartis = px.pie(
                                values=cont_quartis.values,
                                names=cont_quartis.index,
                                title="Segmentação por Quartis & Outliers (Critério IQR)",
                                template="plotly_dark",
                                hole=0.35,
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            fig_quartis.update_traces(textposition="inside", textinfo="percent+label")
                            fig_quartis.update_layout(height=420)
                            st.plotly_chart(fig_quartis, use_container_width=True)

                        st.markdown("##### 📋 Tabela Resumo de Quartis & Limites de Outlier")
                        df_resumo_q = pd.DataFrame({
                            "Parâmetro Estatístico": [
                                "Q1 (25% dos dados)",
                                "Q2 (Mediana / 50%)",
                                "Q3 (75% dos dados)",
                                "IQR (Intervalo Interquartil)",
                                "Limite Inferior (Q1 - 1.5×IQR)",
                                "Limite Superior (Q3 + 1.5×IQR)",
                                "Total de Outliers IQR",
                                "% de Outliers IQR"
                            ],
                            "Valor Calculado": [
                                formatar_valor(estat_metric['q1'], prefixo=prefixo_estat),
                                formatar_valor(estat_metric['q2'], prefixo=prefixo_estat),
                                formatar_valor(estat_metric['q3'], prefixo=prefixo_estat),
                                formatar_valor(estat_metric['iqr'], prefixo=prefixo_estat),
                                formatar_valor(estat_metric['limite_inferior'], prefixo=prefixo_estat),
                                formatar_valor(estat_metric['limite_superior'], prefixo=prefixo_estat),
                                f"{estat_metric['outliers_iqr_count']:,}",
                                f"{estat_metric['outliers_iqr_pct']:.2f}%"
                            ]
                        })
                        st.dataframe(df_resumo_q, use_container_width=True, hide_index=True)

                        with st.expander("🔍 Parecer Estatístico & Recomendações Prescritivas"):
                            st.markdown(f"""
                            - **Variabilidade dos Dados:** Coeficiente de variação de **{cv_val:.2f}%** indica **{'alta' if cv_val > 50 else 'moderada' if cv_val > 30 else 'baixa'}** dispersão relativa dos valores de `{col_estat_sel}`.
                            - **Presença de Outliers:** Identificados **{estat_metric['outliers_iqr_count']:,}** registros além dos limites IQR ({estat_metric['outliers_iqr_pct']:.1f}% do total).
                            - **Diferença Média vs Mediana:** A média ({formatar_valor(estat_metric['media'], prefixo=prefixo_estat)}) está {'acima' if estat_metric['media'] > estat_metric['mediana'] else 'abaixo'} da mediana ({formatar_valor(estat_metric['mediana'], prefixo=prefixo_estat)}), indicando assimetria {'positiva (puxada por valores altos)' if estat_metric['media'] > estat_metric['mediana'] else 'negativa'}.
                            """)

                    except Exception as e_estat:
                        st.error(f"Erro ao processar estatísticas de '{col_estat_sel}': {e_estat}")
