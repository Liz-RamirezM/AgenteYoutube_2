# -*- coding: utf-8 -*-

import os

import streamlit as st


# =========================
# 1. CONFIGURACION DE PAGINA
# =========================

st.set_page_config(
    page_title="Las Damitas Histeria · Agente YouTube",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# 2. CREDENCIALES
# =========================

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("No se encontro GOOGLE_API_KEY en Secrets ni en el archivo .env.")
    st.stop()

try:
    has_gcp_secret = "gcp_service_account" in st.secrets
except Exception:
    has_gcp_secret = False

if not has_gcp_secret and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    st.warning(
        "No se encontro gcp_service_account en Secrets ni GOOGLE_APPLICATION_CREDENTIALS. "
        "Si estas en local, el agente intentara usar credenciales ADC de Google."
    )


# =========================
# 3. IMPORTACION DEL AGENTE
# =========================

try:
    from agent import (
        CHANNEL_ID,
        DATASET_ID,
        PROJECT_ID,
        SEGMENTS_TABLE_ID,
        TABLE_NAME,
        get_agent,
        get_retriever,
    )
except Exception as exc:
    st.error("Error al importar el agente desde agent.py.")
    st.exception(exc)
    st.stop()


# =========================
# 4. ESTILOS
# =========================

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
 
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
 
/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif;
}
 
[data-testid="stAppViewContainer"] {
    background: #F0F0F0;
}
 
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 900px;
}
 
#MainMenu, footer, header { display: none !important; }
 
/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #E0E0E0;
    border-radius: 14px;
    padding: 4px 8px;
    margin-bottom: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
[data-testid="stChatMessageContent"] { color: #282828; }
.stMarkdown p { color: #282828; }
 
/* ── INPUT DE CHAT ── */
[data-testid="stChatInput"] {
    border: 1.5px solid #E0E0E0 !important;
    border-radius: 30px !important;
    background: #ffffff !important;
    padding: 4px 8px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #E8001C !important;
    box-shadow: 0 2px 12px rgba(232,0,28,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    color: #282828 !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: #E8001C !important;
    border-radius: 50% !important;
    border: none !important;
    width: 36px !important;
    height: 36px !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    background: #b8001a !important;
}
[data-testid="stChatInputSubmitButton"] svg {
    fill: white !important;
    color: white !important;
}
 
/* ── SIDEBAR BASE ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #E0E0E0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
/* Ocultar botones nativos visualmente pero mantener funcionalidad */
section[data-testid="stSidebar"] div.stButton > button {
    opacity: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
}
 
/* ── INPUT FOOTER ── */
.yt-input-footer {
    text-align: center;
    font-size: 11px;
    color: #999999;
    margin-top: 4px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# 5. RECURSOS CACHEADOS
# =========================

try:
    retriever = get_retriever()
    agent = get_agent()
except Exception as exc:
    st.error(
        "No se pudo inicializar BigQuery. Revisa `gcp_service_account` en "
        "`.streamlit/secrets.toml` o configura credenciales ADC con Google Cloud."
    )
    st.exception(exc)
    st.stop()


# =========================
# 6. SIDEBAR
# =========================

with st.sidebar:
    st.title("Panel del agente")

    st.markdown("### Fuente de datos")
    st.markdown(
        f"""
        **Proyecto:** `{PROJECT_ID}`  
        **Dataset:** `{DATASET_ID}`  
        **Tabla:** `{TABLE_NAME}`  
        **Canal:** `{CHANNEL_ID}`
        """
    )

    st.markdown("---")
    st.markdown("### Probar conexion")

    if st.button("Probar BigQuery", use_container_width=True):
        with st.spinner("Verificando conexion con BigQuery..."):
            try:
                info = retriever.test_connection()
                st.success("Conexion exitosa")
                st.write("Tabla:", info["tabla"])
                st.write("Filas:", info["filas"])
                st.write("Columnas:", info["columnas"])
            except Exception as exc:
                st.error("No se pudo conectar con BigQuery.")
                st.exception(exc)

    st.markdown("---")
    st.markdown("### Indice BigQuery")

    stats = retriever.transcript_segments_stats()
    if stats["existe"]:
        st.success("Tabla de segmentos lista")
        st.caption(f"Videos: {stats['videos']} | Segmentos: {stats['segmentos']}")
        st.caption(f"Actualizado: {stats['actualizado']}")
        if stats.get("embedding_model"):
            st.caption(f"Embedding model: {stats['embedding_model']}")
    else:
        st.warning("Tabla de segmentos no encontrada")
        st.caption("Creala fuera de Streamlit con el script de indexacion.")

    st.caption(f"Tabla destino: `{SEGMENTS_TABLE_ID}`")
    st.code("python scripts/build_transcript_index.py --force", language="powershell")

    st.markdown("---")
    st.markdown("### Preguntas sugeridas")
    st.markdown(
        """
        - En que episodio se hablo de dinero?
        - En que minuto hablaron de familia?
        - Que temas tienen mejor interaccion?
        - Que videos tienen mas views?
        - Que mejorarias del canal?
        - Que videos rindieron peor de lo esperado?
        """
    )

    st.markdown("---")
    if st.button("Limpiar conversacion", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================
# 7. ENCABEZADO
# =========================

st.markdown(
    '<div class="main-title">Agente Inteligente para YouTube</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Consulta metricas, videos, temas y transcripciones del canal. Para preguntas de
    "en que episodio se hablo de X", el agente usa busqueda semantica por segmentos.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
        <b>Modo recomendado</b><br>
        <span class="small-text">
        Crea la tabla de segmentos una vez con el script offline. Despues Streamlit solo
        vectoriza la pregunta del usuario y consulta BigQuery.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# 8. MEMORIA DE CONVERSACION
# =========================

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hola. Puedo ayudarte a analizar videos, metricas, temas, "
                "transcripciones y recomendaciones del canal."
            ),
        }
    ]


# =========================
# 9. HISTORIAL
# =========================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================
# 10. CHAT
# =========================

prompt = st.chat_input("Ej: En que episodio se hablo de productividad?")

if prompt:
    history_for_agent = st.session_state.messages[-8:]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando BigQuery y transcripciones..."):
            try:
                answer = agent.answer(
                    prompt,
                    history=history_for_agent,
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                message = (
                    "**Ocurrio un error al procesar tu pregunta.**\n\n"
                    f"`{str(exc)}`\n\n"
                    "Revisa Secrets, permisos de BigQuery y que la tabla de segmentos exista "
                    "si estas preguntando por momentos dentro de transcripciones."
                )
                st.error(message)
                st.exception(exc)
                st.session_state.messages.append({"role": "assistant", "content": message})
