# -*- coding: utf-8 -*-
"""
App Streamlit – Previsão de processos judiciais de tramitação LENTA (DataJud/CNJ – TJSP)

Aluno: Fábio Carlos Lima
Curso: Tecnologia em Ciência de Dados – FATEC Adamantina
Atividade: Desenvolvimento do modelo de ML/Otimização

Como executar:
    pip install -r requirements.txt
    streamlit run app_processos_lentos.py
"""

import io
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Processos lentos – TJSP", page_icon="⚖️", layout="wide")

SEMENTE = 42
np.random.seed(SEMENTE)

API_KEY_PADRAO = "cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw=="
ENDPOINT = "https://api-publica.datajud.cnj.jus.br/api_publica_tjsp/_search"

CLASSES = {
    7: "Procedimento Comum Cível",
    436: "Procedimento do Juizado Especial Cível",
    159: "Execução de Título Extrajudicial",
    1116: "Execução Fiscal",
}
MOV_BAIXA = {22, 246}  # 22 = Baixa Definitiva | 246 = Arquivamento Definitivo

VAR_CATEGORICAS = ["classe_processual", "assunto", "orgao_julgador", "municipio_ibge",
                   "sistema", "formato"]
VAR_NUMERICAS = ["qtd_assuntos", "nivel_sigilo", "mes_ajuizamento", "dia_semana_ajuizamento"]

GRADES = {
    "Regressão Logística": {"modelo__C": [0.01, 0.1, 1, 10],
                            "modelo__class_weight": [None, "balanced"]},
    "Random Forest": {"modelo__n_estimators": [200, 400],
                      "modelo__max_depth": [None, 10, 20],
                      "modelo__min_samples_leaf": [1, 5]},
    "Gradient Boosting": {"modelo__n_estimators": [100, 200],
                          "modelo__learning_rate": [0.05, 0.1],
                          "modelo__max_depth": [2, 3]},
}

DIAS_SEMANA = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]


# ---------------------------------------------------------------------------
# Funções – Extract / Transform
# ---------------------------------------------------------------------------
def extrair_classe(codigo_classe, api_key, paginas, tamanho, log, tentativas=3):
    """Extrai processos do 1º grau de uma classe processual, com paginação e nova tentativa."""
    headers = {"Authorization": f"APIKey {api_key}", "Content-Type": "application/json"}
    consulta = {
        "size": tamanho,
        "query": {"bool": {"must": [
            {"match": {"classe.codigo": codigo_classe}},
            {"match": {"grau": "G1"}},
        ]}},
        "sort": [{"@timestamp": {"order": "asc"}}],
    }
    registros = []
    for pagina in range(paginas):
        hits = None
        for tentativa in range(1, tentativas + 1):
            try:
                resp = requests.post(ENDPOINT, headers=headers, json=consulta, timeout=120)
                resp.raise_for_status()
                hits = resp.json().get("hits", {}).get("hits", [])
                break
            except requests.exceptions.HTTPError as erro:
                codigo = erro.response.status_code if erro.response is not None else "?"
                if codigo in (429, 502, 503, 504) and tentativa < tentativas:
                    espera = 5 * tentativa
                    log(f"  [AVISO] HTTP {codigo} – nova tentativa em {espera}s ({tentativa}/{tentativas})")
                    time.sleep(espera)
                    continue
                log(f"  [ERRO] A API retornou HTTP {codigo}. Se for 401/403, atualize a API_KEY.")
                break
            except requests.exceptions.RequestException as erro:
                log(f"  [ERRO] Falha de conexão com a API: {erro}")
                break
        if not hits:
            break
        registros.extend(h["_source"] for h in hits)
        consulta["search_after"] = hits[-1]["sort"]
        log(f"  classe {codigo_classe} – página {pagina + 1}: {len(hits)} processos")
        time.sleep(1)
    return registros


def converter_data(valor):
    """Converte as datas do DataJud (ISO ou AAAAMMDDhhmmss) para datetime sem fuso."""
    if valor is None or valor == "":
        return pd.NaT
    try:
        texto = str(valor)
        if texto.isdigit() and len(texto) >= 8:
            return pd.to_datetime(texto[:14].ljust(14, "0"), format="%Y%m%d%H%M%S", errors="coerce")
        return pd.to_datetime(texto, errors="coerce", utc=True).tz_localize(None)
    except Exception:
        return pd.NaT


def nome(obj):
    return obj.get("nome") if isinstance(obj, dict) else obj


def transformar(dados_brutos):
    """Transforma o JSON bruto em um DataFrame tabular (uma linha por processo)."""
    linhas = []
    for src in dados_brutos:
        try:
            movimentos = src.get("movimentos") or []
            datas_baixa = [converter_data(m.get("dataHora")) for m in movimentos
                           if m.get("codigo") in MOV_BAIXA]
            datas_baixa = [d for d in datas_baixa if pd.notna(d)]
            assuntos = src.get("assuntos") or []
            assunto_principal = nome(assuntos[0]) if assuntos and isinstance(assuntos[0], dict) else None
            orgao = src.get("orgaoJulgador") or {}
            linhas.append({
                "id_processo": src.get("numeroProcesso"),
                "tribunal": src.get("tribunal"),
                "grau": src.get("grau"),
                "orgao_julgador": nome(orgao),
                "municipio_ibge": str(orgao.get("codigoMunicipioIBGE")) if isinstance(orgao, dict) else None,
                "classe_processual": nome(src.get("classe")),
                "assunto": assunto_principal,
                "qtd_assuntos": len(assuntos),
                "data_ajuizamento": converter_data(src.get("dataAjuizamento")),
                "sistema": nome(src.get("sistema")),
                "formato": nome(src.get("formato")),
                "nivel_sigilo": src.get("nivelSigilo", 0),
                "data_baixa": min(datas_baixa) if datas_baixa else pd.NaT,
            })
        except Exception:
            continue
    return pd.DataFrame(linhas).drop_duplicates(subset="id_processo")


def limpar_e_criar_alvo(df_completo):
    """Mantém processos encerrados, calcula a duração e cria a variável alvo."""
    df = df_completo.dropna(subset=["data_ajuizamento", "data_baixa"]).copy()
    df["duracao_dias"] = (df["data_baixa"] - df["data_ajuizamento"]).dt.days
    df = df[(df["duracao_dias"] > 0) & (df["duracao_dias"] <= 365 * 20)]
    df = df[df["data_ajuizamento"].dt.year >= 2010]
    df["mes_ajuizamento"] = df["data_ajuizamento"].dt.month
    df["dia_semana_ajuizamento"] = df["data_ajuizamento"].dt.dayofweek
    mediana = df["duracao_dias"].median()
    df["lento"] = (df["duracao_dias"] > mediana).astype(int)
    return df, mediana


@st.cache_data(show_spinner=False)
def extrair_e_transformar(api_key, classes_escolhidas, paginas, tamanho):
    mensagens = []
    dados_brutos = []
    for codigo in classes_escolhidas:
        mensagens.append(f"Extraindo: {CLASSES[codigo]} ({codigo})")
        dados_brutos.extend(extrair_classe(codigo, api_key, paginas, tamanho, mensagens.append))
    mensagens.append(f"Total extraído: {len(dados_brutos)} processos")
    df_completo = transformar(dados_brutos) if dados_brutos else pd.DataFrame()
    return df_completo, mensagens


def ler_csv(arquivo):
    """Lê o CSV gerado pela pipeline (dataset_ml_processos_tjsp.csv)."""
    # id_processo (número CNJ, 20 dígitos) é lido como texto: como número ele passa do
    # limite de inteiros de 64 bits e quebra a exibição da tabela (OverflowError no pyarrow)
    df = pd.read_csv(arquivo, encoding="utf-8-sig",
                     dtype={"id_processo": str, "municipio_ibge": str})
    for col in ["data_ajuizamento", "data_baixa"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Funções – Modelagem
# ---------------------------------------------------------------------------
def criar_pipeline(modelo):
    preprocessador = ColumnTransformer([
        ("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)),
        ]), VAR_CATEGORICAS),
        ("num", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), VAR_NUMERICAS),
    ])
    return Pipeline([("preprocessamento", preprocessador), ("modelo", modelo)])


def novos_modelos():
    return {
        "Baseline (Dummy)": DummyClassifier(strategy="most_frequent"),
        "Regressão Logística": LogisticRegression(max_iter=2000, random_state=SEMENTE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEMENTE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEMENTE),
    }


def preparar_xy(df):
    X = df[VAR_CATEGORICAS + VAR_NUMERICAS].copy()
    X[VAR_CATEGORICAS] = X[VAR_CATEGORICAS].astype(str)
    y = df["lento"]
    return X, y


@st.cache_resource(show_spinner=False)
def treinar_tudo(df, otimizar):
    """Executa toda a pipeline de ML e devolve um dicionário com os resultados."""
    X, y = preparar_xy(df)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEMENTE)

    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    metricas_cv = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    modelos = novos_modelos()

    resultados = []
    for nome_modelo, modelo in modelos.items():
        cv = cross_validate(criar_pipeline(modelo), X_treino, y_treino,
                            cv=validacao, scoring=metricas_cv)
        resultados.append({"modelo": nome_modelo,
                           **{m: cv[f"test_{m}"].mean() for m in metricas_cv}})
    comparacao = pd.DataFrame(resultados).set_index("modelo").round(3)

    melhor_nome = comparacao.drop("Baseline (Dummy)")["f1"].idxmax()
    if otimizar:
        busca = GridSearchCV(criar_pipeline(modelos[melhor_nome]), GRADES[melhor_nome],
                             scoring="f1", cv=validacao, n_jobs=-1)
        busca.fit(X_treino, y_treino)
        modelo_final = busca.best_estimator_
        melhores_params = busca.best_params_
        f1_otimizado = busca.best_score_
    else:
        modelo_final = criar_pipeline(modelos[melhor_nome]).fit(X_treino, y_treino)
        melhores_params = {}
        f1_otimizado = comparacao.loc[melhor_nome, "f1"]

    y_pred = modelo_final.predict(X_teste)
    y_prob = modelo_final.predict_proba(X_teste)[:, 1]
    baseline = criar_pipeline(DummyClassifier(strategy="most_frequent")).fit(X_treino, y_treino)
    y_base = baseline.predict(X_teste)

    rotulo = f"{melhor_nome} {'otimizado' if otimizar else '(padrão)'}"
    metricas_teste = pd.DataFrame({
        "Baseline (Dummy)": [accuracy_score(y_teste, y_base),
                             precision_score(y_teste, y_base, zero_division=0),
                             recall_score(y_teste, y_base), f1_score(y_teste, y_base), 0.5],
        rotulo: [accuracy_score(y_teste, y_pred), precision_score(y_teste, y_pred),
                 recall_score(y_teste, y_pred), f1_score(y_teste, y_pred),
                 roc_auc_score(y_teste, y_prob)],
    }, index=["Acurácia", "Precisão", "Recall", "F1-score", "ROC-AUC"]).round(3)

    imp = permutation_importance(modelo_final, X_teste, y_teste, scoring="roc_auc",
                                 n_repeats=10, random_state=SEMENTE, n_jobs=-1)
    importancias = pd.Series(imp.importances_mean, index=X_teste.columns).sort_values()

    return {
        "X_treino": X_treino, "X_teste": X_teste, "y_treino": y_treino, "y_teste": y_teste,
        "comparacao": comparacao, "melhor_nome": melhor_nome, "rotulo": rotulo,
        "melhores_params": melhores_params, "f1_otimizado": f1_otimizado,
        "modelo_final": modelo_final, "y_pred": y_pred, "y_prob": y_prob,
        "metricas_teste": metricas_teste, "importancias": importancias,
        "relatorio": classification_report(y_teste, y_pred,
                                           target_names=["rápido (0)", "lento (1)"], digits=3),
    }


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configurações")
fonte = st.sidebar.radio("Fonte dos dados", ["Extrair da API DataJud", "Carregar CSV já gerado"])

if fonte == "Extrair da API DataJud":
    api_key = st.sidebar.text_input("API_KEY pública do CNJ", value=API_KEY_PADRAO, type="password",
                                    help="Se der erro 401/403, copie a chave atual em "
                                         "datajud-wiki.cnj.jus.br/api-publica/acesso")
    classes_escolhidas = st.sidebar.multiselect(
        "Classes processuais", options=list(CLASSES), default=list(CLASSES),
        format_func=lambda c: f"{c} – {CLASSES[c]}")
    paginas = st.sidebar.slider("Páginas por classe", 1, 10, 3)
    tamanho = st.sidebar.select_slider("Registros por página", [100, 250, 500, 1000], value=1000)
    botao_extrair = st.sidebar.button("🔎 Extrair dados", type="primary")
else:
    arquivo_csv = st.sidebar.file_uploader("dataset_ml_processos_tjsp.csv", type="csv")

otimizar = st.sidebar.checkbox("Otimizar com GridSearchCV", value=True,
                               help="Mais lento, mas busca os melhores hiperparâmetros.")

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.title("⚖️ Previsão de processos de tramitação lenta – TJSP")
st.caption("Fábio Carlos Lima · Tecnologia em Ciência de Dados – FATEC Adamantina · "
           "Dados: API Pública DataJud (CNJ)")

with st.expander("📌 Contexto e problema de negócio", expanded=False):
    st.markdown("""
**É possível prever, no momento do ajuizamento, se um processo terá tramitação lenta?**

* **Variável alvo (`lento`)**: 1 se o tempo entre o ajuizamento e a **baixa definitiva** for maior
  que a mediana dos processos analisados; 0 caso contrário.
* **Variáveis explicativas**: só informações conhecidas no início do processo (classe, assunto, vara,
  comarca, sistema, formato, sigilo, mês e dia da semana do ajuizamento), para evitar *data leakage*.
* O ano de ajuizamento **não** é usado: como só entram processos encerrados, processos antigos tendem
  a ter durações maiores por construção.
""")

# ---------------------------------------------------------------------------
# Carregamento dos dados
# ---------------------------------------------------------------------------
if fonte == "Extrair da API DataJud":
    if botao_extrair:
        if not classes_escolhidas:
            st.warning("Escolha pelo menos uma classe processual.")
            st.stop()
        with st.spinner("Extraindo dados da API DataJud… isso pode levar alguns minutos."):
            df_completo, mensagens = extrair_e_transformar(api_key, tuple(classes_escolhidas),
                                                           paginas, tamanho)
        st.session_state["df_completo"] = df_completo
        st.session_state["log"] = mensagens
else:
    if arquivo_csv is not None:
        st.session_state["df_completo"] = ler_csv(arquivo_csv)
        st.session_state["log"] = [f"CSV carregado: {arquivo_csv.name}"]

if "df_completo" not in st.session_state or st.session_state["df_completo"].empty:
    st.info("👈 Escolha a fonte dos dados na barra lateral e clique em **Extrair dados** "
            "(ou carregue o CSV gerado pela pipeline).")
    if st.session_state.get("log"):
        st.code("\n".join(st.session_state["log"]))
        st.error("Nenhum processo foi extraído. Verifique a API_KEY e a conexão.")
    st.stop()

df_completo = st.session_state["df_completo"]
df, MEDIANA = limpar_e_criar_alvo(df_completo)

if len(df) < 200:
    st.error(f"Só {len(df)} processos encerrados – poucos para treinar. "
             "Aumente as páginas por classe.")
    st.stop()

# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------
abas = st.tabs(["📥 Dados", "📊 Análise exploratória", "🤖 Modelos", "🧪 Teste",
                "🔑 Importância", "📝 Interpretação", "🔮 Prever novo processo"])

# --- 1. Dados ---------------------------------------------------------------
with abas[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processos únicos", f"{len(df_completo):,}".replace(",", "."))
    c2.metric("Com baixa definitiva", f"{df_completo['data_baixa'].notna().sum():,}".replace(",", "."))
    c3.metric("Dataset final", f"{len(df):,}".replace(",", "."))
    c4.metric("Mediana da duração", f"{MEDIANA:.0f} dias", f"~{MEDIANA / 365:.1f} anos",
              delta_color="off")

    if st.session_state.get("log"):
        with st.expander("Log da extração"):
            st.code("\n".join(st.session_state["log"]))

    st.subheader("Dataset de Machine Learning")
    # colunas de texto convertidas para str por segurança na exibição
    df_exibir = df.copy()
    for col in df_exibir.select_dtypes(include="object").columns:
        df_exibir[col] = df_exibir[col].astype(str)
    st.dataframe(df_exibir, use_container_width=True, height=350)
    st.download_button("⬇️ Baixar CSV (dataset_ml_processos_tjsp.csv)",
                       df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="dataset_ml_processos_tjsp.csv", mime="text/csv")

# --- 2. EDA -----------------------------------------------------------------
with abas[1]:
    st.subheader("Estatísticas descritivas")
    col_a, col_b = st.columns(2)
    col_a.dataframe(df[["duracao_dias", "qtd_assuntos"]].describe().round(1))
    col_b.dataframe(df["lento"].value_counts(normalize=True)
                    .rename({0: "0 - rápido", 1: "1 - lento"}).round(3).to_frame("proporção"))

    fig, eixos = plt.subplots(1, 3, figsize=(18, 4.5))
    sns.histplot(df["duracao_dias"] / 365, bins=40, ax=eixos[0], color="#3b6ea5")
    eixos[0].axvline(MEDIANA / 365, color="red", linestyle="--",
                     label=f"mediana = {MEDIANA / 365:.1f} anos")
    eixos[0].set_title("Duração até a baixa definitiva")
    eixos[0].set_xlabel("anos"); eixos[0].legend()
    sns.countplot(x=df["lento"].map({0: "rápido", 1: "lento"}), ax=eixos[1],
                  palette=["#5a9e6f", "#c0504d"], order=["rápido", "lento"])
    eixos[1].set_title("Variável alvo"); eixos[1].set_xlabel("")
    df.groupby("classe_processual")["lento"].mean().sort_values().plot(
        kind="barh", ax=eixos[2], color="#3b6ea5")
    eixos[2].set_title("% de processos lentos por classe"); eixos[2].set_xlabel("proporção")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Duração (dias) por sistema e formato")
    st.dataframe(df.groupby(["sistema", "formato"])["duracao_dias"]
                 .agg(["count", "median"]).round(0))

# --- Treinamento (executado uma vez e guardado em cache) ---------------------
with st.spinner("Treinando e avaliando os modelos (validação cruzada"
                + (" + GridSearchCV" if otimizar else "") + ")…"):
    R = treinar_tudo(df, otimizar)

# --- 3. Modelos -------------------------------------------------------------
with abas[2]:
    st.subheader("Comparação dos modelos – validação cruzada 5-fold (treino)")
    st.caption(f"Treino: {len(R['X_treino'])} processos · Teste: {len(R['X_teste'])} processos "
               "(divisão 80/20 estratificada)")
    st.dataframe(R["comparacao"].style.highlight_max(axis=0, color="#d4edda"),
                 use_container_width=True)
    st.bar_chart(R["comparacao"][["accuracy", "f1", "roc_auc"]])

    st.subheader("Otimização de hiperparâmetros")
    st.write(f"Modelo escolhido: **{R['melhor_nome']}**")
    if otimizar:
        st.json(R["melhores_params"])
        c1, c2 = st.columns(2)
        c1.metric("F1 antes", f"{R['comparacao'].loc[R['melhor_nome'], 'f1']:.3f}")
        c2.metric("F1 depois", f"{R['f1_otimizado']:.3f}",
                  f"{R['f1_otimizado'] - R['comparacao'].loc[R['melhor_nome'], 'f1']:+.3f}")
    else:
        st.info("GridSearchCV desativado na barra lateral – usando os hiperparâmetros padrão.")

# --- 4. Teste ---------------------------------------------------------------
with abas[3]:
    st.subheader("Avaliação final no conjunto de teste (20%)")
    st.dataframe(R["metricas_teste"], use_container_width=True)
    st.code(R["relatorio"])

    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay.from_predictions(R["y_teste"], R["y_pred"],
                                            display_labels=["rápido", "lento"],
                                            cmap="Blues", ax=eixos[0])
    eixos[0].set_title("Matriz de confusão – teste")
    RocCurveDisplay.from_predictions(R["y_teste"], R["y_prob"], ax=eixos[1], name=R["melhor_nome"])
    eixos[1].plot([0, 1], [0, 1], "k--", label="acaso (AUC = 0,5)")
    eixos[1].set_title("Curva ROC – teste"); eixos[1].legend()
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("O que cada métrica mede"):
        st.markdown("""
| Métrica | O que mede |
|---|---|
| **Acurácia** | % total de acertos |
| **Precisão** | dos processos apontados como lentos, quantos realmente eram |
| **Recall** | dos processos realmente lentos, quantos o modelo encontrou |
| **F1-score** | média harmônica entre precisão e recall |
| **ROC-AUC** | capacidade de separar lentos de rápidos (0,5 = acaso; 1,0 = perfeito) |
""")

# --- 5. Importância ---------------------------------------------------------
with abas[4]:
    st.subheader("Importância por permutação (queda no ROC-AUC)")
    st.caption("Embaralha uma variável por vez no teste e mede quanto o ROC-AUC piora.")
    fig, ax = plt.subplots(figsize=(8, 5))
    R["importancias"].plot(kind="barh", ax=ax, color="#3b6ea5")
    ax.set_xlabel("queda média no ROC-AUC")
    st.pyplot(fig)
    st.dataframe(R["importancias"].sort_values(ascending=False).round(4).to_frame("importância"))

# --- 6. Interpretação -------------------------------------------------------
with abas[5]:
    m = R["metricas_teste"][R["rotulo"]]
    b = R["metricas_teste"]["Baseline (Dummy)"]
    vn, fp, fn, vp = confusion_matrix(R["y_teste"], R["y_pred"]).ravel()
    auc = m["ROC-AUC"]
    nivel = ("excelente" if auc >= 0.9 else "bom" if auc >= 0.8 else "razoável" if auc >= 0.7
             else "fraco" if auc >= 0.6 else "próximo do acaso")
    top3 = ", ".join(R["importancias"].sort_values(ascending=False).index[:3])
    n_teste = len(R["y_teste"])

    st.subheader(f"Interpretação dos resultados – {R['rotulo']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Acurácia", f"{m['Acurácia']:.1%}", f"{(m['Acurácia'] - b['Acurácia']) * 100:+.1f} p.p.")
    c2.metric("Precisão", f"{m['Precisão']:.1%}")
    c3.metric("Recall", f"{m['Recall']:.1%}")
    c4.metric("F1-score", f"{m['F1-score']:.3f}")
    c5.metric("ROC-AUC", f"{auc:.3f}", nivel, delta_color="off")

    st.markdown(f"""
* **Base:** {len(df)} processos encerrados do TJSP; teste com {n_teste} processos.
* **Acurácia de {m['Acurácia']:.1%}:** o modelo acertou {vp + vn} de {n_teste} processos, contra
  {b['Acurácia']:.1%} do baseline (que sempre prevê a classe mais comum).
* **Precisão de {m['Precisão']:.1%}:** dos {vp + fp} processos apontados como LENTOS, {vp} realmente
  eram lentos ({fp} alarmes falsos).
* **Recall de {m['Recall']:.1%}:** dos {vp + fn} processos realmente lentos, o modelo identificou
  {vp} e deixou passar {fn}.
* **ROC-AUC de {auc:.3f}:** poder de separação **{nivel}** entre processos lentos e rápidos.
* **Variáveis mais importantes:** {top3}.
""")
    if auc >= 0.7:
        st.success("O modelo supera claramente o baseline e identifica padrões de lentidão a partir "
                   "de informações disponíveis no ajuizamento.")
    else:
        st.warning("O modelo supera o baseline, mas o poder preditivo é limitado: a duração também "
                   "depende de fatores ausentes nos metadados (complexidade, recursos, atuação das partes).")

    st.markdown("""
**Limitações:** só metadados públicos e processos já encerrados (viés de seleção – processos em
andamento, muitas vezes os mais longos, não entram).

**Próximos passos:** mais tribunais e classes, movimentos iniciais como variáveis, análise de
sobrevivência e integração do modelo à DAG do Airflow.
""")

# --- 7. Previsão de novo processo -------------------------------------------
with abas[6]:
    st.subheader("Simular a previsão para um novo processo")
    X_ref = R["X_treino"]

    def opcoes(coluna):
        return X_ref[coluna].value_counts().index.tolist()

    with st.form("form_previsao"):
        c1, c2 = st.columns(2)
        entrada = {
            "classe_processual": c1.selectbox("Classe processual", opcoes("classe_processual")),
            "assunto": c1.selectbox("Assunto principal", opcoes("assunto")),
            "orgao_julgador": c1.selectbox("Órgão julgador (vara)", opcoes("orgao_julgador")),
            "municipio_ibge": c1.selectbox("Município (código IBGE)", opcoes("municipio_ibge")),
            "sistema": c2.selectbox("Sistema", opcoes("sistema")),
            "formato": c2.selectbox("Formato", opcoes("formato")),
            "qtd_assuntos": c2.number_input("Quantidade de assuntos", 1, 20, 1),
            "nivel_sigilo": c2.number_input("Nível de sigilo", 0, 5, 0),
            "mes_ajuizamento": c2.selectbox("Mês do ajuizamento", list(range(1, 13))),
            "dia_semana_ajuizamento": c2.selectbox("Dia da semana do ajuizamento", list(range(7)),
                                                   format_func=lambda d: DIAS_SEMANA[d]),
        }
        enviar = st.form_submit_button("Prever", type="primary")

    if enviar:
        novo = pd.DataFrame([entrada])[VAR_CATEGORICAS + VAR_NUMERICAS]
        novo[VAR_CATEGORICAS] = novo[VAR_CATEGORICAS].astype(str)
        prob = R["modelo_final"].predict_proba(novo)[0, 1]
        if prob >= 0.5:
            st.error(f"⏳ Previsão: **LENTO** – probabilidade de {prob:.1%} "
                     f"de passar de {MEDIANA:.0f} dias até a baixa definitiva.")
        else:
            st.success(f"⚡ Previsão: **RÁPIDO** – probabilidade de ser lento: {prob:.1%}.")
        st.progress(float(prob))

    st.divider()
    st.subheader("Exemplos do conjunto de teste")
    exemplo = R["X_teste"].head(10).copy()
    exemplo["prob_lento"] = R["modelo_final"].predict_proba(exemplo)[:, 1].round(3)
    exemplo["previsao"] = np.where(exemplo["prob_lento"] >= 0.5, "lento", "rápido")
    st.dataframe(exemplo[["classe_processual", "assunto", "orgao_julgador", "prob_lento", "previsao"]],
                 use_container_width=True)

    buffer = io.BytesIO()
    joblib.dump(R["modelo_final"], buffer)
    st.download_button("⬇️ Baixar modelo treinado (modelo_processos_lentos.joblib)",
                       buffer.getvalue(), file_name="modelo_processos_lentos.joblib")
