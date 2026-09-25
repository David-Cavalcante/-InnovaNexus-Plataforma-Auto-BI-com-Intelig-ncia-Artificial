import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração do backend 'Agg' do Matplotlib antes de importar o pyplot (obrigatório para ambientes como o Streamlit)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
import google.generativeai as genai
from io import StringIO
import os
import tempfile
from fpdf import FPDF

# ---------------------------------------------------------
# CONFIGURAÇÃO DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="InnovaNexus - Insights & Dashboards com IA",
    page_icon="🚀",
    layout="wide"
)

# ---------------------------------------------------------
# CONFIGURAÇÃO FIXA DA CHAVE DE API DO GEMINI
# ---------------------------------------------------------
API_KEY_REAL = "AQ.Ab8RN6JH8xPlqQuW1oz08gaZMr3GAblLDYoVuYIz3zHjY6609g"
API_KEY_INTERNA = API_KEY_REAL if API_KEY_REAL != "COLE_SUA_CHAVE_GEMINI_AQUI" else os.environ.get("GEMINI_API_KEY", "")

MESES_PT = {
    'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março',
    'April': 'Abril', 'May': 'Maio', 'June': 'Junho',
    'July': 'Julho', 'August': 'Agosto', 'September': 'Setembro',
    'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
}

# ---------------------------------------------------------
# GERENCIAMENTO DE SESSÃO & AUTENTICAÇÃO (LOGIN)
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
# FUNÇÃO DE GERAÇÃO DE PDF EXECUTIVO (FPDF2 + MATPLOTLIB)
# ---------------------------------------------------------
class RelatorioPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, 'InnovaNexus - Relatório Executivo de Dados', border=False, new_x="LMARGIN", new_y="NEXT", align='L')
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, 'Parecer Prescritivo de Negócio, Diagnóstico de ETL e Visualização de Indicadores', border=False, new_x="LMARGIN", new_y="NEXT", align='L')
        self.ln(5)
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Página {self.page_no()} | Plataforma InnovaNexus Auto-BI', align='C')

def limpar_texto_pdf(texto):
    """Substitui caracteres especiais e sanitiza codificação para Latin-1 (Helvetica)."""
    substituicoes = {
        '•': '-', '–': '-', '—': '-',
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '🚀': '', '🎯': '', '📌': '', '💡': '', '✨': '', '🧠': '', '🔬': '', '👀': '', '📊': '', '⚠️': 'ALERTA:'
    }
    for orig, subst in substituicoes.items():
        texto = texto.replace(orig, subst)
    return texto.encode('latin-1', 'replace').decode('latin-1')

def gerar_pdf_executivo(total_linhas, total_colunas, nulos_totais, parecer_texto, df, var_num, var_cat):
    pdf = RelatorioPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 1. Resumo Técnico de ETL
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, '1. RESUMO DE ENGENHARIA E ESTRUTURA DOS DADOS', new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(60, 7, f'- Total de Registros: {total_linhas:,}', border=0)
    pdf.cell(60, 7, f'- Total de Colunas: {total_colunas}', border=0)
    pdf.cell(60, 7, f'- Valores Ausentes: {nulos_totais}', border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # 2. Gerar e Inserir Gráficos no PDF
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, '2. PAINEL VISUAL DE INDICADORES CHAVE', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        
        # Gráfico 1: Barras da métrica principal por categoria
        df_top = df.groupby(var_cat)[var_num].mean().reset_index().sort_values(by=var_num, ascending=False).head(8)
        ax1.bar(df_top[var_cat].astype(str), df_top[var_num], color='#2563eb')
        ax1.set_title(f'Média de {var_num} por {var_cat}', fontsize=8, fontweight='bold')
        ax1.tick_params(axis='x', rotation=35, labelsize=6)
        ax1.tick_params(axis='y', labelsize=6)
        
        # Gráfico 2: Pizza da distribuição das maiores categorias
        ax2.pie(df_top[var_num], labels=df_top[var_cat].astype(str), autopct='%1.1f%%', textprops={'fontsize': 6})
        ax2.set_title(f'Distribuição Proporcional', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name, dpi=150)
            pdf.image(tmpfile.name, x=10, y=pdf.get_y(), w=190)
            pdf.ln(68)
        plt.close()
    except Exception as e:
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 8, limpar_texto_pdf(f'(Gráficos omitidos: {e})'), new_x="LMARGIN", new_y="NEXT")
    
    # 3. Parecer da IA / Recomendações
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, '3. PARECER EXECUTIVO E RECOMENDAÇÕES (INNOVANEXUS AI)', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    pdf.set_font('Helvetica', '', 9.5)
    pdf.set_text_color(51, 65, 85)
    
    texto_processado = parecer_texto.replace('#', '').replace('*', '').replace('`', '')
    texto_limpo = limpar_texto_pdf(texto_processado)
    
    pdf.multi_cell(0, 5, texto_limpo)
    
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
# FUNÇÃO DE INTEGRAÇÃO COM GEMINI AI (ATUALIZADA PARA O NOVO MODELO)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def gerar_insights_gemini(api_key, df_info_str, df_describe_str, df_head_str):
    try:
        from google import genai
        
        client = genai.Client(api_key=api_key)
        
        prompt = (
            "Atue como um Especialista em Analytics e Designer de Relatórios Executivos.\n"
            "Analise a estrutura de dados fornecida abaixo e gere um parecer estruturado estritamente em Português do Brasil de alto nível executivo e estratégico.\n\n"
            "Siga exatamente a seguinte estrutura e estilo em Markdown:\n\n"
            "# InnovaNexus - Relatório Executivo de Dados\n"
            "*Parecer Prescritivo de Negócio, Diagnóstico de ETL e Visualização de Indicadores*\n\n"
            "---\n\n"
            "> **🎯 Objetivo Estratégico:** Monitorar o tempo excessivo de permanência dos veículos parados nas instalações (`facility_id`), identificando gargalos operacionais que impactam diretamente os custos de frota, o nível de serviço ao cliente e a receita potencial em risco.\n\n"
            "---\n\n"
            "## 📊 1. Indicadores-Chave de Desempenho (KPIs) & Fórmulas\n\n"
            "Apresente uma tabela limpa e formatada exatamente com as colunas: 'Métrica', 'Cálculo / Fórmula' e 'Objetivo Estratégico'.\n"
            "Inclua os seguintes KPIs calculados com base nas colunas do dataset:\n"
            "- Nível de Pontualidade (On-Time Delivery)\n"
            "- Eficiência Operacional (Atraso Médio)\n"
            "- Eficiência de Pátio (Tempo de Detenção)\n"
            "- Variabilidade de Atraso (Desvio Padrão)\n"
            "- Receita Potencial em Risco (Revenue-at-Risk)\n\n"
            "---\n\n"
            "## 💡 2. Insights Estratégicos da Operação\n\n"
            "Inclua uma caixa de destaque (blockquote '>') com o Alerta Crítico apontando a taxa média de pontualidade real observada na base.\n\n"
            "Destaque em subseções numeradas os 3 diagnósticos principais:\n"
            "1. **Crise de Pontualidade:** Análise da porcentagem de entregas no prazo e o impacto na satisfação do cliente.\n"
            "2. **Gargalos de Carga e Descarga:** Análise do tempo médio de detenção (minutos_detencao) e picos de retenção.\n"
            "3. **Alta Volatilidade Operacional:** Análise do desvio padrão dos atrasos (atraso_real_minutos) e oscilações da malha.\n\n"
            "---\n\n"
            "## 🚀 3. Ações Prescritivas para a Gestão\n\n"
            "Forneça 3 recomendações acionáveis direcionadas para a tomada de decisão da diretoria:\n"
            "1. **Auditoria Direcionada de Instalações e Rotas (Princípio de Pareto 80/20):** Foco nos 20% de locais (facility_id / estado_local) que geram 80% dos atrasos.\n"
            "2. **Recalibragem Dinâmica das Janelas de Agendamento (Dynamic Scheduling):** Reajuste dos horários agendados (data_hora_agendada) frente à realidade do trânsito.\n"
            "3. **Plano de Retenção de Clientes Estratégicos:** Cruzamento da receita_anual_potencial com a pontualidade por cliente para atendimento prioritário.\n\n"
            "--- DADOS PARA ANÁLISE ---\n"
            f"Tipos de Colunas e Nulos:\n{df_info_str}\n\n"
            f"Resumo Estatístico:\n{df_describe_str}\n\n"
            f"Amostra da Base:\n{df_head_str}\n"
        )
        
        # Tentativa com o modelo atualizado indicado pelo erro da API
        erros_tentados = []
        for modelo in ['gemini-3.8-flash', 'gemini-1.5-flash']:
            try:
                response = client.models.generate_content(
                    model=modelo,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text, None
            except Exception as ex:
                erros_tentados.append(f"{modelo}: {str(ex)}")
                continue

        return None, f"Erro na API do Gemini. Detalhes: {' | '.join(erros_tentados)}"

    except Exception as e:
        return None, f"Erro ao inicializar o client Gemini: {str(e)}"
# ---------------------------------------------------------
# TELA DE LOGIN (QUANDO NÃO AUTENTICADO)
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

        st.caption("Acesso demonstrativo para a banca Samsung Innovation Campus.")

# ---------------------------------------------------------
# SISTEMA PRINCIPAL (APÓS LOGIN)
# ---------------------------------------------------------
else:
    st.title("🚀 InnovaNexus: Insights & Dashboards com IA")
    st.caption("Plataforma para ETL automático, pareceres prescritivos com IA e visualização executiva de dados.")

    st.sidebar.header("👤 Sessão do Utilizador")
    st.sidebar.write("Conectado como: **Administrador**")
    st.sidebar.success("🤖 Motor de IA (Gemini): **Ativo**")

    if st.sidebar.button("Sair (Logout)"):
        realizar_logout()

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Fonte de Dados")
    fonte_dados = st.sidebar.radio(
        "Escolha a fonte de entrada de dados:",
        ["Upload de Arquivo (.csv)", "Conexão com Banco de Dados (SQL)"]
    )

    df = None

    if fonte_dados == "Upload de Arquivo (.csv)":
        st.subheader("📁 Upload de Arquivo CSV")
        arquivo_carregado = st.file_uploader("Selecione um ficheiro CSV para análise", type=["csv"], key="uploader_csv_innovanexus")

        if arquivo_carregado is not None:
            try:
                df = pd.read_csv(arquivo_carregado)

                if df.shape[1] == 1 and ',' in str(df.columns[0]):
                    arquivo_carregado.seek(0)
                    linhas = [line.decode('utf-8').strip() for line in arquivo_carregado.readlines()]
                    linhas_limpas = [l[1:-1] if l.startswith('"') and l.endswith('"') else l for l in linhas]
                    df = pd.read_csv(StringIO('\n'.join(linhas_limpas)))

                df = executar_etl_automatico(df)
                st.success(f"Ficheiro CSV carregado e processado via ETL com sucesso! ({df.shape[1]} colunas e métricas identificadas)")

            except Exception as e:
                st.error(f"Erro ao ler e processar o ficheiro CSV: {e}")

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
                        st.success("Dados carregados e limpos via ETL a partir da base de dados!")
                        st.session_state['df_sql'] = df_sql
            else:
                st.warning("Por favor, informe o nome da tabela.")

        if 'df_sql' in st.session_state and fonte_dados == "Conexão com Banco de Dados (SQL)":
            df = st.session_state['df_sql']

    if df is not None:
        st.markdown("---")

        tab_dados, tab_diagnostico, tab_ia, tab_dash = st.tabs([
            "👀 Visão Geral dos Dados",
            "🔬 Diagnóstico Técnico",
            "🤖 Parecer da IA & Insights",
            "📊 Dashboard Interativo"
        ])

        info_df = pd.DataFrame({
            "Coluna": df.columns,
            "Tipo de Dado": [str(dtype) for dtype in df.dtypes],
            "Valores Ausentes": df.isnull().sum().values,
            "% Ausentes": (df.isnull().sum().values / len(df) * 100).round(2)
        })

        # ABA 1: VISÃO GERAL
        with tab_dados:
            st.subheader("Pré-visualização da Tabela (Pós-ETL & Métricas Criadas)")
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            col_kpi1.metric("Total de Linhas", f"{df.shape[0]:,}")
            col_kpi2.metric("Total de Colunas", df.shape[1])
            col_kpi3.metric("Valores Ausentes Totais", int(df.isnull().sum().sum()))

            num_linhas = st.slider("Quantidade de linhas a exibir na tabela:", min_value=10, max_value=100, value=25, step=5)
            st.dataframe(df.head(num_linhas), use_container_width=True)

        # ABA 2: DIAGNÓSTICO TÉCNICO
        with tab_diagnostico:
            st.subheader("Análise Automática da Estrutura de Dados")
            col_diag1, col_diag2 = st.columns(2)

            with col_diag1:
                st.markdown("**Tipos de Colunas e Valores Ausentes**")
                st.dataframe(info_df, use_container_width=True)

            with col_diag2:
                st.markdown("**Resumo Estatístico (Colunas Numéricas)**")
                if not df.select_dtypes(include=['number']).empty:
                    st.dataframe(df.describe().T, use_container_width=True)
                else:
                    st.info("Não foram encontradas colunas numéricas na base de dados.")

        # ABA 3: PARECER DA IA & DOWNLOAD DE PDF
        with tab_ia:
            st.subheader("🧠 Parecer Executivo Automático (InnovaNexus AI)")

            if not API_KEY_INTERNA:
                st.error("⚠️ Nenhuma Chave de API do Gemini foi configurada no sistema.")
            else:
                btn_gerar_ia = st.button("✨ Gerar Parecer & Insights Prescritivos")

                if btn_gerar_ia:
                    with st.spinner("A IA do InnovaNexus está a analisar a estrutura dos dados e a formular recomendações..."):
                        info_str = info_df.to_string(index=False)
                        describe_str = df.describe().T.to_string() if not df.select_dtypes(include=['number']).empty else "Nenhuma variável numérica."
                        head_str = df.head(5).to_string(index=False)

                        insights, erro_ia = gerar_insights_gemini(API_KEY_INTERNA.strip(), info_str, describe_str, head_str)

                        if erro_ia:
                            st.error(f"Erro ao gerar parecer com a IA: {erro_ia}")
                        else:
                            st.session_state['insights_ia'] = insights

                if 'insights_ia' in st.session_state:
                    st.markdown(st.session_state['insights_ia'])

                    st.markdown("---")
                    st.subheader("📄 Exportar Relatório Executivo")

                    col_num = df.select_dtypes(include=['number']).columns.tolist()
                    col_cat = [c for c in df.select_dtypes(include=['object', 'category']).columns.tolist() if 'id' not in c.lower()]

                    if not col_num or not col_cat:
                        st.warning("A base necessita de ter pelo menos uma coluna numérica e uma categórica para gerar o PDF.")
                    else:
                        var_num_pdf = 'atraso_real_minutos' if 'atraso_real_minutos' in col_num else col_num[0]
                        var_cat_pdf = 'tipo_cliente' if 'tipo_cliente' in col_cat else col_cat[0]

                        try:
                            pdf_data = gerar_pdf_executivo(
                                df.shape[0],
                                df.shape[1],
                                int(df.isnull().sum().sum()),
                                st.session_state['insights_ia'],
                                df,
                                var_num_pdf,
                                var_cat_pdf
                            )

                            st.download_button(
                                label="📥 Baixar Relatório Executivo Completo em PDF",
                                data=pdf_data,
                                file_name="InnovaNexus_Relatorio_Executivo.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.error(f"Erro ao gerar o PDF: {e}")

        # ABA 4: DASHBOARD INTERATIVO
        with tab_dash:
            st.subheader("📊 Painel de Visualização Dinâmico e Limpo")

            col_num = df.select_dtypes(include=['number']).columns.tolist()
            col_cat = [c for c in df.select_dtypes(include=['object', 'category']).columns.tolist() if 'id' not in c.lower()]
            if not col_cat:
                col_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()

            col_dates = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()

            if not col_num:
                st.warning("A base de dados não contém colunas numéricas suficientes para gerar os gráficos estatísticos.")
            elif not col_cat:
                st.warning("A base de dados não contém colunas categóricas para os agrupamentos.")
            else:
                st.markdown("##### 🎛️ Filtros e Seleção de Variáveis")
                c1, c2, c3, c4 = st.columns(4)

                idx_num_default = col_num.index('atraso_real_minutos') if 'atraso_real_minutos' in col_num else 0
                idx_cat_default = col_cat.index('tipo_cliente') if 'tipo_cliente' in col_cat else 0

                with c1:
                    var_num = st.selectbox("Métrica Numérica (Eixo Y):", col_num, index=idx_num_default)
                with c2:
                    var_cat = st.selectbox("Categoria Principal (Eixo X):", col_cat, index=idx_cat_default)
                with c3:
                    var_cor = st.selectbox("Agrupamento por Cor (Opcional):", ["Nenhuma"] + col_cat)
                with c4:
                    top_n = st.slider("Limite de Categorias Exibidas:", min_value=5, max_value=20, value=10)

                st.markdown("---")

                g1, g2 = st.columns(2)

                with g1:
                    st.markdown(f"**Top {top_n} Média de `{var_num}` por `{var_cat}`**")
                    color_param = None if var_cor == "Nenhuma" else var_cor

                    df_grouped = df.groupby(var_cat if color_param is None else [var_cat, color_param])[var_num].mean().reset_index()
                    df_top = df_grouped.sort_values(by=var_num, ascending=False).head(top_n)

                    fig_bar = px.bar(
                        df_top,
                        x=var_cat,
                        y=var_num,
                        color=color_param,
                        text_auto='.1f',
                        labels={var_cat: var_cat, var_num: f"Média de {var_num}"},
                        title=f"Top {top_n} {var_cat} por {var_num}",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with g2:
                    st.markdown(f"**Distribuição Proporcional de `{var_num}`**")
                    df_pie = df.groupby(var_cat)[var_num].sum().reset_index().sort_values(by=var_num, ascending=False).head(8)
                    fig_pie = px.pie(
                        df_pie,
                        names=var_cat,
                        values=var_num,
                        hole=0.4,
                        labels={var_cat: "Categoria", var_num: "Valor Total"},
                        title="Distribuição das Top 8 Categorias",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("---")
                if col_dates:
                    st.markdown("##### 📅 Evolução e Tendência Temporal")
                    var_date = st.selectbox("Selecione a Coluna de Data para Tendência:", col_dates)

                    df_time = df.set_index(var_date).resample('D')[var_num].mean().reset_index()
                    fig_line = px.line(
                        df_time,
                        x=var_date,
                        y=var_num,
                        labels={var_date: "Data", var_num: f"Média de {var_num}"},
                        title=f"Evolução Diária da Média de {var_num}",
                        markers=True,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.markdown("##### 🔍 Análise de Correlação e Dispersão")
                    outras_metricas = [c for c in col_num if c != var_num]
                    if outras_metricas:
                        var_num2 = st.selectbox("Segunda Métrica Numérica (Eixo X):", outras_metricas)
                        fig_scatter = px.scatter(
                            df,
                            x=var_num2,
                            y=var_num,
                            color=None if var_cor == "Nenhuma" else var_cor,
                            hover_data=col_cat[:2],
                            labels={var_num2: var_num2, var_num: var_num},
                            title=f"Relação entre {var_num2} e {var_num}",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
