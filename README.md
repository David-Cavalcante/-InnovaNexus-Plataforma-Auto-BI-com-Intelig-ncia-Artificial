# 🚀 InnovaNexus: Insights & Dashboards com IA

## > **Projeto em Andamento — Desenvolvido para o Samsung Innovation Campus**  
> *Plataforma inteligente para ingestão automatizada de dados (ETL), diagnósticos analíticos, pareceres prescritivos com IA (Google Gemini) e relatórios executivos dinâmicos.*
# 📌 Sobre o Projeto
O InnovaNexus é uma solução corporativa de Business Intelligence as a Service (BIaaS) projetada para democratizar a análise de dados. A plataforma elimina barreiras técnicas ao automatizar todo o pipeline de engenharia de dados (ETL), permitindo que gestores e analistas transformem bases brutas (CSV ou Bancos de Dados SQL) em painéis interativos e relatórios executivos gerados por Inteligência Artificial em segundos.

Este projeto está em desenvolvimento ativo como parte da entrega para o Samsung Innovation Campus, refletindo padrões modernos de arquitetura de software, engenharia de dados e integração com LLMs de última geração.

# 🛠️ Arquitetura & Stack Tecnológica
O ecossistema do InnovaNexus foi construído com tecnologias robustas e de alto desempenho:

Interface & Web Framework: Streamlit (Arquitetura reativa e multi-abas com painéis interativos).

Engenharia de Dados (ETL): Pandas & SQLAlchemy (Tratamento automatizado de nulos, tipagem estrita, conversão de datas e conexões relacionais).

Inteligência Artificial: Google Generative AI (Gemini) com descoberta dinâmica de modelos (gemini-1.5-flash / gemini-pro) para geração de pareceres prescritivos e KPIs estratégicos.

Visualização Gráfica: Plotly Express (Dashboards interativos) e Matplotlib (Backend Agg otimizado para relatórios estáticos).

Exportação Corporativa: FPDF2 (Geração automatizada de relatórios executivos em PDF com sanitização de caracteres e inserção de gráficos).

## ✨ Principais Funcionalidades
## 🔐 Portal de Acesso Executivo: Sistema de autenticação integrado com controle de sessão.

🔌 Conectividade Agnóstica: Suporte a arquivos CSV com tratamento de aspas e formatações irregulares, além de conexão nativa com bancos de dados relacionais (SQLite, PostgreSQL, MySQL).

## ⚙️ Pipeline de ETL Automatizado:

Limpeza de duplicatas e padronização de strings.

Detecção e conversão automática de colunas temporais (com suporte a localização de meses em Português do Brasil).

Engenharia de recursos nativa (cálculo automático de atrasos em minutos, métricas de pontualidade e tratamento inteligente de valores ausentes).

## 🔬 Diagnóstico Técnico: Visão estrutural detalhada com matriz de tipos de dados, contagem de nulos e resumo estatístico descritivo.

## 🤖 InnovaNexus AI (Parecer Prescritivo): Análise contextual profunda realizada pelo Gemini, estruturando diagnósticos de negócio, tabelas de KPIs e planos de ação práticos baseados no Princípio de Pareto.

📊 Dashboard Interativo Dinâmico: Filtros customizáveis, seleções de métricas numéricas e categóricas (Eixo X/Y), gráficos de barras, pizza, dispersão e séries temporais.

📄 Exportação de Relatórios em PDF: Criação instantânea de relatórios executivos contendo sumário de engenharia, gráficos gerados em tempo de execução e o parecer da IA formatado.

⚙️ Como Executar o Projeto Localmente
Siga os passos abaixo para configurar e rodar a aplicação no seu ambiente de desenvolvimento:

## 1. Clonar o Repositório
Bash
git clone [https://github.com/seu-usuario/InnovaNexus.git](https://github.com/seu-usuario/InnovaNexus.git)
cd InnovaNexus
2. Instalar as Dependências
Certifique-se de ter o Python instalado e execute:

Bash
pip install -r requirements.txt
3. Configurar o Ambiente
O projeto conta com chaves de integração otimizadas, mas você pode definir variáveis de ambiente caso necessário ou utilizar a configuração embutida no código.

4. Executar a Aplicação Streamlit
No terminal, execute o comando:

Bash
python -m streamlit run app.py
Acesse no seu navegador através de: http://localhost:8501

🔑 Credenciais de Acesso de Demonstração (Banca)
Para acessar o portal executivo na tela de login:

Usuário: admin

Senha: 1234

📈 Status do Projeto
[x] Arquitetura base e sistema de autenticação

[x] Pipeline de ETL e adaptação agnóstica para datasets

[x] Integração avançada com Google Gemini AI e cache de performance

[x] Dashboard dinâmico com Plotly e exportação de PDF corporativo

[ ] Implementação de novas fontes de dados em nuvem (Em desenvolvimento)

# 👤 Autor
## Desenvolvido por David Cavalcante

Projeto acadêmico e profissional desenvolvido no âmbito do programa Samsung Innovation Campus.
