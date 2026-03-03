import os
import tempfile
import uuid
import soundfile as sf
import streamlit as st
from pydub import AudioSegment
from pydub.effects import normalize

# Importa a poderosa API do F5-TTS
from f5_tts.api import F5TTS

# =======================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =======================================================================
st.set_page_config(page_title="GenVox - F5 Studio", page_icon="🎙️", layout="centered")

# =======================================================================
# CACHE DO MODELO (Evita recarregar a cada clique)
# =======================================================================
@st.cache_resource(show_spinner="Carregando o motor F5-TTS (Pode demorar um pouco na primeira vez)...")
def load_model():
    # O F5-TTS automaticamente acha a GPU e baixa os pesos necessários
    return F5TTS()

f5tts = load_model()

# =======================================================================
# FRONT-END
# =======================================================================
st.title("🎙️ GenVox - F5 Studio")
st.markdown("O seu estúdio na nuvem focado em **Clonagem de Voz Ultrarrealista**.")

with st.container():
    st.subheader("1. A Voz Original (Clone)")
    arquivo_clone = st.file_uploader("Upload do Áudio de Referência (O ideal é de 10 a 15 segundos)", type=["wav", "mp3", "ogg"])
    
    texto_referencia = st.text_area(
        "O que a pessoa diz no áudio acima? (Opcional)", 
        placeholder="Se deixar em branco, a IA vai transcrever automaticamente para você..."
    )

st.divider()

st.subheader("2. O Roteiro")
texto = st.text_area("Escreva o que a nova voz deverá falar:", height=150)

st.divider()

if st.button("🚀 CLONAR E GERAR ÁUDIO", type="primary", use_container_width=True):
    if not texto.strip():
        st.error("⚠️ Por favor, digite o roteiro que a voz deverá falar.")
    elif not arquivo_clone:
        st.error("⚠️ Por favor, faça o upload do áudio que será clonado.")
    else:
        with st.spinner("Processando Inteligência Artificial na GPU..."):
            temp_dir = tempfile.gettempdir()
            caminho_ref = os.path.join(temp_dir, f"ref_{uuid.uuid4()}.wav")
            
            # Salva o upload temporariamente na máquina
            with open(caminho_ref, "wb") as f:
                f.write(arquivo_clone.getbuffer())
                
            try:
                # O motor mágico do F5-TTS faz todo o trabalho duro aqui
                wav, sr, _ = f5tts.infer(
                    ref_file=caminho_ref, 
                    ref_text=texto_referencia.strip(), 
                    gen_text=texto.strip()
                )
                
                caminho_saida = os.path.join(temp_dir, f"saida_{uuid.uuid4()}.wav")
                sf.write(caminho_saida, wav, sr)
                
                # Tratamento de áudio com Pydub (sua lógica original de estúdio)
                audio_final = AudioSegment.from_wav(caminho_saida)
                if len(audio_final) > 0:
                    audio_final = audio_final.high_pass_filter(180)
                    audio_final = normalize(audio_final, headroom=3.0)
                    audio_final.export(caminho_saida, format="wav")
                    
                    st.success("✅ Áudio clonado com perfeição!")
                    st.audio(caminho_saida, format="audio/wav")
                else:
                    st.error("🚨 O áudio gerado está vazio. Tente outro arquivo de referência.")
                    
            except Exception as e:
                st.error(f"Erro durante a geração: {str(e)}")
