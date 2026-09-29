import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json

# Configuração do backend 'Agg' do Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from openai import OpenAI
from io import StringIO
import os
import tempfile
from fpdf import FPDF
import numpy as np

# ---------------------------------------------------------
# CONFIGURAÇÃO DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="InnovaNexus - Insights & Dashboards com IA",
    page_icon="🚀",
    layout="wide"
)

# CSS personalizado
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

# ---------------------------------------------------------
# CHAVES PADRÃO DO SISTEMA
# ⚠️  ATENÇÃO: As chaves abaixo são EXEMPLOS — substitua pelas suas!
#
#   Groq    → https://console.groq.com/keys          (começa com "gsk_")
#   Gemini  → https://aistudio.google.com/apikey     (começa com "AIza")
#   DeepSeek→ https://platform.deepseek.com/         (começa com "sk-")
#
# Deixe o campo vazio ("") para forçar o utilizador a inserir a chave na sidebar.
# ---------------------------------------------------------
CHAVES_PADRAO = {
    "Groq (Rápido/Gratuito)": "",   # ← Cole aqui sua chave Groq  (gsk_...)
    "Gemini (Google)":         "",   # ← Cole aqui sua chave Gemini (AIza...)
    "DeepSeek (Avançado)":     "",   # ← Cole aqui sua chave DeepSeek (sk-...)
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
        '•': '-', '–': '-', '—': '-', '"': '"', '"': '"', ''': "'", ''': "'",
        '🚀': '', '🎯': '', '📌': '', '💡': '', '✨': '', '🧠': '', '🔬': '',
        '👀': '', '📊': '', '⚠️': 'ALERTA:', '📈': '', '📉': '', '🔑': '',
        '✅': '', '❌': '', '📋': '', '🏆': '', '💰': '', '📅': ''
    }
    for orig, subst in substituicoes.items():
        texto = texto.replace(orig, subst)
    return texto.encode('latin-1', 'replace').decode('latin-1')

def gerar_pdf_executivo(total_linhas, total_colunas, nulos_totais, parecer_texto, df, var_num, var_cat):
    pdf = RelatorioPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, '1. RESUMO DE ENGENHARIA E ESTRUTURA DOS DADOS', new_x="LMARGIN", new_y="NEXT")

    pdf.set_font('Helvetica', '', 10)
    pdf.cell(60, 7, f'- Total de Registros: {total_linhas:,}', border=0)
    pdf.cell(60, 7, f'- Total de Colunas: {total_colunas}', border=0)
    pdf.cell(60, 7, f'- Valores Ausentes: {nulos_totais}', border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, '2. PAINEL VISUAL DE INDICADORES CHAVE', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        df_top = df.groupby(var_cat)[var_num].mean().reset_index().sort_values(by=var_num, ascending=False).head(8)
        ax1.bar(df_top[var_cat].astype(str), df_top[var_num], color='#2563eb')
        ax1.set_title(f'Media de {var_num} por {var_cat}', fontsize=8, fontweight='bold')
        ax1.tick_params(axis='x', rotation=35, labelsize=6)
        ax1.tick_params(axis='y', labelsize=6)

        ax2.pie(df_top[var_num], labels=df_top[var_cat].astype(str), autopct='%1.1f%%', textprops={'fontsize': 6})
        ax2.set_title('Distribuicao Proporcional', fontsize=8, fontweight='bold')

        plt.tight_layout()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name, dpi=150)
            pdf.image(tmpfile.name, x=10, y=pdf.get_y(), w=190)
            pdf.ln(68)
        plt.close()
    except Exception as e:
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 8, limpar_texto_pdf(f'(Graficos omitidos: {e})'), new_x="LMARGIN", new_y="NEXT")

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
# FUNÇÃO DE ETL AUTOMÁTICO
# ---------------------------------------------------------
def executar_etl_automatico(df_raw):
    df_clean = df_raw.copy()
    df_clean = df_clean.drop_duplicates()

    colunas_texto = df_clean.select_dtypes(include=['object']).columns
    for col in colunas_texto:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    for col in df_clean.columns:
        if any(k in col.lower() for k in ('date', 'time', 'dt', 'data', 'hora')):
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            except Exception:
                pass

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

    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].ffill()
            else:
                df_clean[col] = df_clean[col].fillna("Não informado")

    colunas_datas = df_clean.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns
    for col in colunas_datas:
        prefixo = col.replace('_datetime', '').replace('_date', '').replace('_data', '')
        df_clean[f'{prefixo}_ano'] = df_clean[col].dt.year
        df_clean[f'{prefixo}_mes'] = df_clean[col].dt.month_name().map(MESES_PT).fillna(df_clean[col].dt.month_name())

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
# INTEGRAÇÃO COM IA MULTIMOTOR — VERSÃO CORRIGIDA
# ---------------------------------------------------------
def gerar_insights_ia(motor, api_key, df_info_str, df_describe_str, df_head_str):
    """
    Gera insights via IA usando o motor selecionado.

    Correções aplicadas:
    - Groq: modelo atualizado para 'llama-3.3-70b-versatile' (llama3-8b-8192 foi descontinuado)
    - Gemini: usa REST API com modelo 'gemini-1.5-flash' via endpoint público
    - DeepSeek: tratamento de erros melhorado
    """
    try:
        info_safe    = df_info_str[:1500]
        desc_safe    = df_describe_str[:2500]
        head_safe    = df_head_str[:1200]

        prompt = (
            "Atue como um Especialista em Analytics e Designer de Relatórios Executivos.\n"
            "Analise a estrutura de dados fornecida abaixo e gere um parecer estruturado "
            "estritamente em Português do Brasil de alto nível executivo e estratégico.\n\n"
            "Siga exatamente a seguinte estrutura e estilo em Markdown:\n\n"
            "# InnovaNexus - Relatório Executivo de Dados\n"
            "*Parecer Prescritivo de Negócio, Diagnóstico de ETL e Visualização de Indicadores*\n\n"
            "---\n\n"
            "> **🎯 Objetivo Estratégico:** Monitorar gargalos operacionais que impactam "
            "diretamente os custos, o nível de serviço ao cliente e a receita em risco.\n\n"
            "---\n\n"
            "## 📊 1. Indicadores-Chave de Desempenho (KPIs)\n\n"
            "Apresente uma tabela markdown com os KPIs extraídos ou inferidos da base de dados.\n\n"
            "---\n\n"
            "## 💡 2. Insights Estratégicos da Operação\n\n"
            "Destaque em subseções numeradas os 3 diagnósticos principais encontrados.\n\n"
            "---\n\n"
            "## 🚀 3. Ações Prescritivas para a Gestão\n\n"
            "Forneça 3 recomendações acionáveis para a tomada de decisão da diretoria.\n\n"
            "--- DADOS PARA ANÁLISE ---\n"
            f"Tipos de Colunas:\n{info_safe}\n\n"
            f"Resumo Estatístico:\n{desc_safe}\n\n"
            f"Amostra de Dados:\n{head_safe}\n"
        )

        system_msg = "Você é um analista de dados sênior especialista em business intelligence."

        # ── 1. MOTOR GROQ ─────────────────────────────────────────────────────
        if motor == "Groq (Rápido/Gratuito)":
            # Modelos ativos na Groq em 2025 (em ordem de preferência)
            # Fonte: https://console.groq.com/docs/models
            modelos_groq = [
                "llama-3.3-70b-versatile",      # Llama 3.3 — recomendado
                "llama-3.1-8b-instant",         # Llama 3.1 rápido
                "llama3-8b-8192",               # Llama 3 clássico
                "gemma2-9b-it",                 # Google Gemma 2
                "mixtral-8x7b-32768",           # Mixtral (pode estar descontinuado)
            ]
            client = OpenAI(
                api_key=api_key.strip(),
                base_url="https://api.groq.com/openai/v1"
            )
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
                        stream=False
                    )
                    return response.choices[0].message.content, None
                except Exception as e:
                    err_str = str(e)
                    ultimo_erro = err_str
                    # Erro de autenticação — chave inválida, interrompe imediatamente
                    if "401" in err_str or "authentication" in err_str.lower() or "invalid_api_key" in err_str.lower():
                        return None, (
                            "❌ Chave Groq inválida ou expirada (erro 401).\n\n"
                            "Sua chave pode ter expirado. Obtenha uma nova em:\n"
                            "➡️ https://console.groq.com/keys\n\n"
                            "A chave começa com **gsk_** — cole-a no campo 'Chave de API' na sidebar."
                        )
                    # Modelo descontinuado — tenta o próximo
                    if any(t in err_str.lower() for t in ["decommissioned", "deprecated", "not found", "model_not_found", "404"]):
                        continue
                    # Outro erro — interrompe
                    break
            return None, (
                f"❌ Erro na API da Groq: {ultimo_erro}\n\n"
                "Verifique sua chave em https://console.groq.com/keys"
            )

        # ── 2. MOTOR GEMINI (SDK google-genai ou REST) ─────────────────────────
        elif motor == "Gemini (Google)":
            key = api_key.strip()
            if not key:
                return None, (
                    "🔑 Chave Gemini não configurada.\n\n"
                    "Obtenha uma chave no Google AI Studio em https://aistudio.google.com/apikey\n"
                    "As chaves atuais começam com **AQ.** — cole-a no campo 'Chave de API' no painel lateral."
                )

            # Tenta via SDK google-genai (preferido)
            try:
                from google import genai as ggenai
                os.environ["GEMINI_API_KEY"] = key  # SDK lê esta variável
                gc = ggenai.Client(api_key=key)

                # Modelos em ordem de preferência
                modelos_sdk = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-1.5-flash", "gemini-2.0-flash"]
                ultimo_erro_sdk = None
                for modelo_sdk in modelos_sdk:
                    try:
                        interaction = gc.interactions.create(
                            model=modelo_sdk,
                            input=f"{system_msg}\n\n{prompt}",
                        )
                        return interaction.output_text, None
                    except Exception as e_sdk:
                        ultimo_erro_sdk = str(e_sdk)
                        if any(t in str(e_sdk).lower() for t in ["not found", "404", "deprecated"]):
                            continue
                        break  # Erro de auth ou rede — não adianta tentar outros modelos

                # Se falhou via SDK, tenta via REST como fallback
                raise Exception(f"SDK falhou em todos os modelos: {ultimo_erro_sdk}")

            except ImportError:
                pass  # SDK não instalada — usa REST abaixo
            except Exception as e_sdk_final:
                # SDK falhou — tenta REST
                pass

            # Fallback: REST API direta
            modelos_rest = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.0-pro"]
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
                        # Verifica se a resposta tem conteúdo
                        candidatos = dados.get("candidates", [])
                        if candidatos:
                            partes = candidatos[0].get("content", {}).get("parts", [])
                            if partes:
                                return partes[0].get("text", ""), None
                        return None, "Resposta vazia do Gemini. Tente novamente."
                    elif resp.status_code in (404, 400):
                        continue  # Tenta próximo modelo
                    elif resp.status_code == 403:
                        return None, (
                            f"❌ Acesso negado (HTTP 403). Verifique se a chave 'AIza...' "
                            "é válida e tem a API 'Generative Language' habilitada no Google AI Studio."
                        )
                    elif resp.status_code == 429:
                        return None, "⏳ Limite de requisições atingido (quota). Aguarde 1 minuto e tente novamente."
                    else:
                        try:
                            msg = resp.json().get('error', {}).get('message', resp.text[:300])
                        except Exception:
                            msg = resp.text[:300]
                        return None, f"Erro HTTP {resp.status_code} da API Gemini: {msg}"
                except requests.exceptions.Timeout:
                    return None, "⏱️ Timeout ao conectar com o Gemini. Verifique sua conexão e tente novamente."
                except requests.exceptions.ConnectionError:
                    return None, "🌐 Sem conexão com a internet ou API do Gemini inacessível."
                except Exception as e_rest:
                    return None, f"Erro inesperado ao chamar o Gemini: {str(e_rest)}"

            return None, (
                "Nenhum modelo Gemini respondeu. "
                "Confirme que sua chave 'AIza...' está ativa em aistudio.google.com."
            )

        # ── 3. MOTOR DEEPSEEK ──────────────────────────────────────────────────
        elif motor == "DeepSeek (Avançado)":
            client = OpenAI(
                api_key=api_key.strip(),
                base_url="https://api.deepseek.com/v1"
            )
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": prompt}
                    ],
                    max_tokens=2048,
                    stream=False
                )
                if response and response.choices:
                    return response.choices[0].message.content, None
                return None, "Resposta vazia da API do DeepSeek."
            except Exception as e:
                erro_str = str(e)
                if "authentication" in erro_str.lower() or "401" in erro_str:
                    return None, "Chave da API DeepSeek inválida. Verifique em platform.deepseek.com."
                return None, f"Erro na API do DeepSeek: {erro_str}"

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
        ["Groq (Rápido/Gratuito)", "Gemini (Google)", "DeepSeek (Avançado)"]
    )

    chave_personalizada = st.sidebar.text_input(
        "Chave de API (Opcional)",
        type="password",
        help="Deixe em branco para usar a chave configurada no sistema para o motor selecionado."
    )

    chave_ativa = chave_personalizada.strip() if chave_personalizada.strip() else CHAVES_PADRAO.get(motor_selecionado, "")

    # Valida se a chave parece válida para o motor
    chave_valida = bool(chave_ativa)
    if motor_selecionado == "Gemini (Google)":
        if chave_valida:
            st.sidebar.success("✅ API Gemini: Pronta")
        else:
            st.sidebar.warning("⚠️ Chave Gemini não configurada. Insira sua chave (ex: AQ... ou AIza...) do Google AI Studio.")
    else:
        if chave_valida:
            st.sidebar.success(f"✅ API {motor_selecionado.split()[0]}: Pronta")
        else:
            st.sidebar.warning(f"⚠️ Chave {motor_selecionado.split()[0]} não configurada.")

    # Exibe dica de onde obter a chave
    links_chave = {
        "Groq (Rápido/Gratuito)":  "https://console.groq.com/keys",
        "Gemini (Google)":          "https://aistudio.google.com/apikey",
        "DeepSeek (Avançado)":      "https://platform.deepseek.com/",
    }
    st.sidebar.caption(f"🔑 [Obter chave {motor_selecionado.split()[0]}]({links_chave[motor_selecionado]})")

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Sair (Logout)"):
        realizar_logout()

    # ── FONTE DE DADOS ─────────────────────────────────────────────────────────
    st.markdown("---")
    fonte_dados = st.radio(
        "Escolha a fonte de entrada de dados:",
        ["Upload de Arquivo (.csv)", "Conexão com Banco de Dados (SQL)"],
        horizontal=True
    )

    df = None

    if fonte_dados == "Upload de Arquivo (.csv)":
        arquivo_carregado = st.file_uploader(
            "Selecione um ficheiro CSV para análise",
            type=["csv"],
            key="uploader_csv"
        )
        if arquivo_carregado is not None:
            try:
                df_raw = pd.read_csv(arquivo_carregado)
                # Tenta re-parsear se a separação ficou errada
                if df_raw.shape[1] == 1 and ',' in str(df_raw.columns[0]):
                    arquivo_carregado.seek(0)
                    linhas = [line.decode('utf-8', errors='replace').strip() for line in arquivo_carregado.readlines()]
                    linhas_limpas = [l[1:-1] if l.startswith('"') and l.endswith('"') else l for l in linhas]
                    df_raw = pd.read_csv(StringIO('\n'.join(linhas_limpas)))
                df = executar_etl_automatico(df_raw)
                st.success(
                    f"✅ Ficheiro carregado e processado via ETL! "
                    f"**{df.shape[0]:,} registros** × **{df.shape[1]} colunas** identificadas."
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
            df = st.session_state['df_sql']

    # ══════════════════════════════════════════════════════════════════════════
    # DASHBOARDS — só renderiza se há dados carregados
    # ══════════════════════════════════════════════════════════════════════════
    if df is not None:
        st.markdown("---")

        # Identificação de colunas por tipo
        col_num = df.select_dtypes(include=['number']).columns.tolist()
        col_cat = [
            c for c in df.select_dtypes(include=['object', 'category']).columns.tolist()
            if 'id' not in c.lower()
        ]
        if not col_cat:
            col_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
        col_data = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()

        info_df = pd.DataFrame({
            "Coluna":          df.columns,
            "Tipo de Dado":    [str(d) for d in df.dtypes],
            "Valores Ausentes":df.isnull().sum().values,
            "% Ausentes":      (df.isnull().sum().values / len(df) * 100).round(1),
            "Valores Únicos":  df.nunique().values,
        })

        tab_dados, tab_diagnostico, tab_ia, tab_dash = st.tabs([
            "👀 Visão Geral dos Dados",
            "🔬 Diagnóstico Técnico",
            "🤖 Parecer da IA & Insights",
            "📊 Dashboard Interativo"
        ])

        # ── TAB 1: VISÃO GERAL ─────────────────────────────────────────────────
        with tab_dados:
            st.subheader("📋 Visão Geral dos Dados")

            # KPIs rápidos no topo
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("📦 Total de Registros",  f"{df.shape[0]:,}")
            k2.metric("📐 Colunas",              f"{df.shape[1]}")
            k3.metric("🔢 Colunas Numéricas",    f"{len(col_num)}")
            k4.metric("🏷️ Colunas Categóricas", f"{len(col_cat)}")
            k5.metric("⚠️ Valores Ausentes",     f"{int(df.isnull().sum().sum()):,}")

            st.markdown("---")

            # Controle de quantas linhas exibir
            n_linhas = st.slider(
                "Número de linhas a exibir:",
                min_value=10,
                max_value=min(500, df.shape[0]),
                value=min(50, df.shape[0]),
                step=10
            )
            st.dataframe(df.head(n_linhas), use_container_width=True, height=500)

            # Permite download do dataset tratado
            csv_tratado = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Baixar Dataset Tratado (CSV)",
                data=csv_tratado,
                file_name="dataset_tratado.csv",
                mime="text/csv"
            )

        # ── TAB 2: DIAGNÓSTICO TÉCNICO ─────────────────────────────────────────
        with tab_diagnostico:
            st.subheader("🔬 Diagnóstico Técnico de Qualidade")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 📋 Estrutura e Qualidade")
                st.dataframe(info_df, use_container_width=True, height=400)

            with c2:
                if col_num:
                    st.markdown("##### 📈 Estatísticas Descritivas")
                    st.dataframe(df[col_num].describe().T.round(4), use_container_width=True, height=400)

            # Mapa de calor de valores ausentes
            if df.isnull().sum().sum() > 0:
                st.markdown("##### 🗺️ Distribuição de Valores Ausentes por Coluna")
                ausentes = df.isnull().sum().reset_index()
                ausentes.columns = ['Coluna', 'Ausentes']
                ausentes = ausentes[ausentes['Ausentes'] > 0].sort_values('Ausentes', ascending=False)
                fig_missing = px.bar(
                    ausentes, x='Coluna', y='Ausentes',
                    title="Colunas com Valores Ausentes",
                    template="plotly_dark",
                    color='Ausentes',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_missing, use_container_width=True)

            # Matriz de correlação
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
            st.subheader(f"🧠 Parecer Executivo Automático via {motor_selecionado.split()[0]}")

            if not chave_valida:
                st.error(
                    f"⚠️ Configure uma Chave de API válida para o motor **{motor_selecionado}** "
                    f"no painel lateral esquerdo."
                )
            else:
                if st.button("✨ Gerar Parecer & Insights Prescritivos", type="primary"):
                    with st.spinner(f"A analisar dados via {motor_selecionado}... Aguarde."):
                        info_str     = info_df.to_string(index=False)
                        describe_str = df[col_num].describe().T.to_string() if col_num else "Nenhuma variável numérica."
                        head_str     = df.head(5).to_string(index=False)

                        insights, erro_ia = gerar_insights_ia(
                            motor_selecionado, chave_ativa, info_str, describe_str, head_str
                        )

                        if erro_ia:
                            st.error(f"❌ Falha ao gerar parecer:\n\n{erro_ia}")
                        else:
                            st.session_state['insights_ia'] = insights

                if 'insights_ia' in st.session_state:
                    st.markdown(st.session_state['insights_ia'])

                    if col_num and col_cat:
                        try:
                            pdf_data = gerar_pdf_executivo(
                                df.shape[0], df.shape[1],
                                int(df.isnull().sum().sum()),
                                st.session_state['insights_ia'],
                                df, col_num[0], col_cat[0]
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
                # Seletores de variáveis
                col_sel1, col_sel2, col_sel3 = st.columns(3)
                var_num  = col_sel1.selectbox("📈 Métrica principal (Eixo Y):", col_num)
                var_cat  = col_sel2.selectbox("🏷️ Categoria (Eixo X):", col_cat)
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

                # ── Linha 1: Barras + Pizza ──────────────────────────────────
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

                # ── Linha 2: Box Plot + Histograma ──────────────────────────
                r2c1, r2c2 = st.columns(2)

                with r2c1:
                    # Box plot: distribuição da métrica por categoria (top N)
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
                    # Histograma
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

                # ── Linha 3: Scatter + Barras Horizontais ────────────────────
                r3c1, r3c2 = st.columns(2)

                with r3c1:
                    if len(col_num) >= 2:
                        # Tenta com linha de tendência (requer statsmodels); fallback sem ela
                        try:
                            import statsmodels  # noqa: F401
                            _trendline = "ols"
                        except ImportError:
                            _trendline = None

                        fig_scatter = px.scatter(
                            df, x=var_num, y=var_num2,
                            color=var_cat if len(df[var_cat].unique()) <= 20 else None,
                            title=f"🔵 Dispersão: '{var_num}' vs '{var_num2}'" + (" (instale statsmodels para linha de tendência)" if not _trendline else ""),
                            template="plotly_dark",
                            trendline=_trendline,
                            opacity=0.7
                        )
                        fig_scatter.update_layout(showlegend=False)
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.info("Scatter plot requer ao menos 2 colunas numéricas.")

                with r3c2:
                    # Barras horizontais — comparação de 2 métricas
                    if len(col_num) >= 2 and var_num2 in df.columns:
                        df_comp = df.groupby(var_cat)[[var_num, var_num2]].mean().reset_index()
                        df_comp = df_comp.sort_values(var_num, ascending=False).head(top_n)
                        df_melt = df_comp.melt(id_vars=var_cat, value_vars=[var_num, var_num2],
                                               var_name="Métrica", value_name="Valor")
                        fig_comp = px.bar(
                            df_melt, y=var_cat, x="Valor", color="Métrica",
                            orientation='h',
                            title=f"📊 Comparativo: '{var_num}' vs '{var_num2}'",
                            template="plotly_dark",
                            barmode="group"
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                    else:
                        # Barras horizontais simples
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

                # ── Linha 4: Série Temporal (se houver datas) + Violino ──────
                r4c1, r4c2 = st.columns(2)

                with r4c1:
                    if col_data:
                        col_dt = col_data[0]
                        df_ts = df[[col_dt, var_num]].dropna().sort_values(col_dt)
                        fig_ts = px.line(
                            df_ts, x=col_dt, y=var_num,
                            title=f"📅 Série Temporal: '{var_num}' ao longo do tempo",
                            template="plotly_dark",
                            markers=True
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)
                    else:
                        # Violin plot
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
                    # Treemap — participação proporcional
                    if len(col_num) >= 1 and len(col_cat) >= 1:
                        df_tree = (
                            df.groupby(var_cat)[var_num]
                            .sum()
                            .reset_index()
                            .sort_values(var_num, ascending=False)
                            .head(min(20, df[var_cat].nunique()))
                        )
                        fig_tree = px.treemap(
                            df_tree,
                            path=[var_cat],
                            values=var_num,
                            title=f"🌳 Treemap: Participação de '{var_num}' por '{var_cat}'",
                            template="plotly_dark",
                            color=var_num,
                            color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig_tree, use_container_width=True)

                # ── Tabela resumo no rodapé ──────────────────────────────────
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
