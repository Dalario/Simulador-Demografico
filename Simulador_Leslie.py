import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador Leslie", layout="wide")
st.title("Simulador Demográfico (Matriz de Leslie)")
st.subheader("""Este projeto apresenta um simulador populacional baseado na Matriz de Leslie, permitindo analisar o comportamento da população ao longo do tempo.
""")
st.write("Cole os vetores como números separados por vírgula (ex: 0,4,3,0).")

# ---------- helpers ---------- 
def build_leslie_matrix(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float) if len(b) > 0 else np.array([])
    n = len(a)
    L = np.zeros((n, n), dtype=float)
    L[0, :] = a
    for i in range(n - 1):
        L[i+1, i] = b[i]
    return L

def project_population(L, X0, steps):
    X0 = np.array(X0, dtype=float)
    history = np.zeros((steps+1, len(X0)), dtype=float)
    history[0,:] = X0
    for k in range(1, steps+1):
        history[k,:] = L.dot(history[k-1,:])
    return history

def dominant_eigen(L):
    vals, vecs = np.linalg.eig(L)
    idx = np.argmax(np.real(vals))
    lam = float(np.real(vals[idx]))
    v = np.real(vecs[:, idx])
    if abs(v.sum()) > 1e-12:
        v = v / v.sum()
    return lam, v

# ---------- inputs ----------
st.sidebar.header("Configurações")

n = st.sidebar.number_input("Número de faixas etárias(n)", min_value=2, max_value=50, value=3, step=1)
steps = st.sidebar.slider("Ciclos temporais(K)", 1, 200, 20)

st.sidebar.markdown("**Insira a_i (n números)** — natalidade (primeira linha)")
default_a = "0,4,3" if n==3 else ",".join(["0"]*n)
a_text = st.sidebar.text_area("a_i (vírgula)", value=default_a)
st.sidebar.markdown("**Insira b_i (n-1 números)** — sobrevivência (subdiagonal)")
default_b = "0.5,0.25" if n==3 else ",".join(["0.9"]*(max(1,n-1)))
b_text = st.sidebar.text_area("b_i (vírgula)", value=default_b)

st.sidebar.markdown("**População inicial de cada faixa X0 (n números)**")
default_x0 = ",".join(["100"]*n)
x0_text = st.sidebar.text_area("X0 (vírgula)", value=default_x0)

# ---------- parse inputs with safe fallback ----------
def parse_list(text, expected_len, fill=0.0):
    try:
        lst = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
    except:
        lst = []
    if len(lst) < expected_len:
        lst = lst + [fill] * (expected_len - len(lst))
    return lst[:expected_len]

a = parse_list(a_text, n, fill=0.0)
b = parse_list(b_text, max(1, n-1), fill=0.0)[:(n-1)]
x0 = parse_list(x0_text, n, fill=0.0)

# validações simples
b = [min(max(val, 0.0), 1.0) for val in b]   # garantir 0 <= b <= 1
a = [max(val, 0.0) for val in a]             # garantir a >= 0

st.sidebar.markdown("Exemplo: para n=3 os padrões são: a = 0,4,3 ; b = 0.5,0.25 ; X0 = 100,100,100")

if st.sidebar.button("Iniciar simulação"):
    L = build_leslie_matrix(a, b)
    st.subheader("Matriz de Leslie (L)")
    st.dataframe(pd.DataFrame(L).round(6))

    history = project_population(L, x0, steps)
    df_history = pd.DataFrame(history, columns=[f"Faixa_{i+1}" for i in range(len(a))])
    df_history.index.name = "Período"

    # autovalor e vetor
    lam, v = dominant_eigen(L)
    st.subheader("Autovalor dominante e vetor estacionário (proporções)")
    st.write(f"λ₁ = {lam:.6f}")
    st.dataframe(pd.DataFrame({"Proporção": v.round(6)}, index=[f"Faixa_{i+1}" for i in range(len(v))]))

    # total por período
    st.subheader("Total por período")
    st.line_chart(df_history.sum(axis=1))

    # evolução por faixa
    st.subheader("Evolução por faixa")
    fig, ax = plt.subplots(figsize=(8,4))
    for col in df_history.columns:
        ax.plot(df_history.index, df_history[col], label=col)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Tabela de resultados")
    st.dataframe(df_history.style.format("{:.2f}"))

    csv = df_history.to_csv(index=True).encode("utf-8")
    st.download_button("Baixar CSV", csv, "projecao.csv", "text/csv")

    # interpretação
    if lam > 1:
        st.success("População tende a crescer (λ₁ > 1).")
    elif abs(lam-1) < 1e-6:
        st.info("População tende a estabilizar (λ₁ ≈ 1).")
    else:
        st.warning("População tende a diminuir (λ₁ < 1).")
else:
    st.info("Preencha os vetores à esquerda e clique em 'Rodar simulação'.")
