import os

# 1. Corrige o erro do Matplotlib no Colab
os.environ['MPLBACKEND'] = 'Agg'

# 2. Aceita automaticamente os termos de uso do modelo XTTS-v2
os.environ["COQUI_TOS_AGREED"] = "1"

import torch

# =======================================================================
# CORREÇÃO DE COMPATIBILIDADE PARA PYTORCH 2.6+
# =======================================================================
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# =======================================================================

import tempfile
import uuid
import re
import streamlit as st
from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import normalize

# =======================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =======================================================================
st.set_page_config(page_title="GenVox - Studio", page_icon="🎙️", layout="centered")

# =======================================================================
# CACHE DO MODELO (Evita recarregar 1.8GB a cada clique)
# =======================================================================
@st.cache_resource(show_spinner="Carregando o motor XTTS-v2 (Isso só acontece na primeira vez)...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    vozes = list(tts.synthesizer.tts_model.speaker_manager.name_to_id)
    return tts, vozes

# Carrega o modelo de fato
tts, vozes_padrao = load_model()

# =======================================================================
# FUNÇÕES DE PROCESSAMENTO
# =======================================================================
def limpar_texto_para_ia(texto):
    texto = texto.strip()
    texto = texto.replace("...", ",")
    texto = re.sub(r'\s+', ' ', texto)
    if texto.endswith("."):
        texto = texto[:-1]
    return texto

def fatiar_texto(texto):
    frases = re.split(r'(?<=[.!?])\s+', texto)
    return [f.strip() for f in frases if f.strip()]

# =======================================================================
# FRONT-END STREAMLIT
# =======================================================================
st.title("🎙️ GenVox - Studio")
st.markdown("Bem-vindo ao GenVox. Escolha o modo de operação, configure a voz, insira seu texto e gere o áudio!")

st.divider()

# 1. Modo de Geração
st.subheader("1. Modo de Geração")
modo = st.radio("Escolha como deseja gerar a voz:", ["Clonar Voz", "Usar Voz Padrão"], horizontal=True)

arquivo_clone = None
voz_selecionada = None

# Interface Dinâmica (O Streamlit esconde os campos nativamente com o "if")
if modo == "Clonar Voz":
    arquivo_clone = st.file_uploader("Upload de Áudio de Referência (Formato WAV)", type=["wav", "mp3", "ogg"])
else:
    voz_selecionada = st.selectbox("Selecione a Voz Padrão do Modelo", vozes_padrao)

st.divider()

# 2. Roteiro
st.subheader("2. Roteiro")
texto = st.text_area("Escreva o que a voz deverá falar aqui...", height=150)

# 3. Ajustes Finos
with st.expander("3. Ajustes Finos ⚙️"):
    temperatura = st.slider("Temperatura (Aleatoriedade)", 0.0, 1.0, 0.75, 0.01)
    velocidade = st.slider("Velocidade", 0.5, 2.0, 1.0, 0.01)
    top_p = st.slider("Top-P (Fidelidade)", 0.1, 1.0, 0.95, 0.01)

st.divider()

# Botão de Geração
if st.button("🚀 GERAR ÁUDIO", type="primary", use_container_width=True):
    
    # Validações antes de começar
    if not texto.strip():
        st.error("⚠️ O texto não pode estar vazio. Por favor, digite um roteiro.")
    elif modo == "Clonar Voz" and arquivo_clone is None:
        st.error("⚠️ Por favor, faça o upload de um arquivo de áudio de referência.")
    elif modo == "Usar Voz Padrão" and not voz_selecionada:
        st.error("⚠️ Por favor, selecione uma voz padrão na lista.")
    else:
        # Tudo certo, vamos processar!
        with st.spinner("Processando áudio com a GPU..."):
            temp_dir = tempfile.gettempdir()
            
            # Se for clone, salva o arquivo enviado na memória do Colab temporariamente
            caminho_audio_clone = None
            if arquivo_clone is not None:
                caminho_audio_clone = os.path.join(temp_dir, f"clone_ref_{uuid.uuid4()}.wav")
                with open(caminho_audio_clone, "wb") as f:
                    f.write(arquivo_clone.getbuffer())

            params = {
                "language": "pt",
                "temperature": float(temperatura),
                "speed": float(velocidade),
                "top_p": float(top_p),
                "split_sentences": False 
            }

            if modo == "Clonar Voz":
                params["speaker_wav"] = caminho_audio_clone
            else:
                params["speaker"] = voz_selecionada

            # Inicia o fatiamento e geração
            chunks = fatiar_texto(texto)
            arquivos_de_audio_gerados = []
            
            # Barra de progresso visual do Streamlit
            barra_progresso = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                chunk_final = limpar_texto_para_ia(chunk)
                if not chunk_final: continue
                
                temp_name = f"genvox_part_{uuid.uuid4()}.wav"
                temp_path = os.path.join(temp_dir, temp_name)
                
                try:
                    tts.tts_to_file(text=chunk_final, file_path=temp_path, **params)
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        arquivos_de_audio_gerados.append(temp_path)
                except Exception as e:
                    st.error(f"Erro na parte {i+1}: {e}")
                
                # Atualiza a barra
                barra_progresso.progress((i + 1) / len(chunks))

            # União dos áudios
            audio_final = AudioSegment.empty()
            crossfade_duration = 50
            
            for i, arquivo in enumerate(arquivos_de_audio_gerados):
                try:
                    segmento = AudioSegment.from_wav(arquivo)
                    if i == 0: 
                        audio_final = segmento
                    else: 
                        audio_final = audio_final.append(segmento, crossfade=crossfade_duration)
                except Exception as e:
                    pass

            audio_final = audio_final.high_pass_filter(180)
            audio_final = normalize(audio_final, headroom=3.0)

            caminho_saida_wav = os.path.join(temp_dir, "genvox_saida_stream.wav")
            audio_final.export(caminho_saida_wav, format="wav")

            # Faxina
            for arquivo in arquivos_de_audio_gerados:
                try: os.remove(arquivo)
                except: pass
            if caminho_audio_clone:
                try: os.remove(caminho_audio_clone)
                except: pass

            st.success("✅ Áudio gerado com perfeição!")
            
            # Toca o áudio na tela e já embute o botão nativo de baixar!
            st.audio(caminho_saida_wav, format="audio/wav")
