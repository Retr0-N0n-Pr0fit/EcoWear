import streamlit as st
import pandas as pd
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="EcoWear AI Agent", page_icon="🤖")
st.title("🤖🍃 EcoWear Innovations - AI Agent")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("⚠️ Error: No se encontró la clave GEMINI_API_KEY en los secretos de Streamlit.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

@st.cache_data
def cargar_base_conocimiento():
    textos_contexto = []
    
    archivo_pdf = "Politicas_Sostenibilidad_EcoWear.pdf"
    if os.path.exists(archivo_pdf):
        loader = PyPDFLoader(archivo_pdf)
        paginas = loader.load()
        for p in paginas:
            textos_contexto.append(f"[Fuente: PDF, Página: {p.metadata.get('page', 0) + 1}] {p.page_content}")
            
    archivos_csv = ["stack_ecowear.csv", "ventas_ecowear.csv"]
    for ruta_csv in archivos_csv:
        if os.path.exists(ruta_csv):
            nombre = os.path.basename(ruta_csv)
            df = pd.read_csv(ruta_csv)
            for idx, row in df.iterrows():
                fila = ", ".join([f"{col}: {val}" for col, val in row.items()])
                textos_contexto.append(f"[Fuente: {nombre}, Fila: {idx+1}] {fila}")
                
    return "\n\n".join(textos_contexto)

contexto_unificado = cargar_base_conocimiento()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

system_prompt = (
    "Eres el AI Agent oficial de EcoWear Innovations S.A. (Edición 2026).\n"
    "Responde de manera profesional, amable y concisa basándote estrictamente en el contexto.\n"
    "Si no sabes algo, di que no dispones de esa información.\n\n"
    "Contexto de la empresa:\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "¡Hola! Bienvenido a EcoWear Innovations. Soy tu asistente de Inteligencia Artificial 🤖🍃. ¿En qué puedo ayudarte hoy respecto a nuestro stack tecnológico, reportes de ventas o políticas de sostenibilidad?"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if usuario_input := st.chat_input("Escribe tu consulta para EcoWear aquí..."):
    
    st.session_state.messages.append({"role": "user", "content": usuario_input})
    with st.chat_message("user"):
        st.write(usuario_input)

    with st.chat_message("assistant"):
        mensajes_finales = prompt_template.format_messages(context=contexto_unificado, input=usuario_input)
        respuesta = llm.invoke(mensajes_finales)
        
        st.write(respuesta.content)
        
    st.session_state.messages.append({"role": "assistant", "content": respuesta.content})
