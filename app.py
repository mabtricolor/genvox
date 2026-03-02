import os

# 1. Corrige o erro do Matplotlib no Colab (ANTES de qualquer outra importação)
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
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import normalize

print("Inicializando o ambiente...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo selecionado: {device}")

print("Carregando o modelo XTTS-v2... Isso pode demorar um pouco.")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Extrai a lista de vozes do modelo (lógica original)
vozes_padrao = list(tts.synthesizer.tts_model.speaker_manager.name_to_id)
print("Modelo carregado com sucesso!")

# =======================================================================
# FUNÇÕES DE PROCESSAMENTO DE TEXTO E ÁUDIO (SUA LÓGICA DE VOLTA)
# =======================================================================

def limpar_texto_para_ia(texto):
    texto = texto.strip()
    texto = texto.replace("...", ",")
    # Remove espaços duplos bizarros
    texto = re.sub(r'\s+', ' ', texto)
    # Remove o ponto final na última palavra para a IA não ler "ponto"
    if texto.endswith("."):
        texto = texto[:-1]
    return texto

def fatiar_texto(texto):
    # Substitui o NLTK: Divide o texto onde houver ponto, exclamação ou interrogação, seguido de espaço
    frases = re.split(r'(?<=[.!?])\s+', texto)
    return [f.strip() for f in frases if f.strip()]

def gerar_audio(modo, arquivo_clone, voz_selecionada, texto, temperatura, velocidade, top_p):
    if not texto or not texto.strip():
        raise gr.Error("O texto não pode estar vazio. Por favor, digite um roteiro.")

    # Parâmetros - AVISO: split_sentences=False para não deixar a IA processar o texto!
    params = {
        "language": "pt",
        "temperature": float(temperatura),
        "speed": float(velocidade),
        "top_p": float(top_p),
        "split_sentences": False 
    }

    if modo == "Clonar Voz":
        if not arquivo_clone: raise gr.Error("Por favor, faça o upload de um áudio de referência.")
        params["speaker_wav"] = arquivo_clone
    else:
        if not voz_selecionada: raise gr.Error("Por favor, selecione uma voz padrão.")
        params["speaker"] = voz_selecionada

    temp_dir = tempfile.gettempdir()
    
    # 1. Fatia o texto
    chunks = fatiar_texto(texto)
    arquivos_de_audio_gerados = []
    
    print(f"Texto dividido inteligentemente em {len(chunks)} frases. Gerando...")

    # 2. Gera os áudios separadamente (igual ao seu código desktop)
    for i, chunk in enumerate(chunks):
        chunk_final = limpar_texto_para_ia(chunk)
        if not chunk_final: continue
        
        print(f"Gerando parte {i+1}/{len(chunks)}...")
        temp_name = f"genvox_part_{uuid.uuid4()}.wav"
        temp_path = os.path.join(temp_dir, temp_name)
        
        try:
            tts.tts_to_file(text=chunk_final, file_path=temp_path, **params)
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                arquivos_de_audio_gerados.append(temp_path)
        except Exception as e:
            print(f"Erro na geração da parte {i+1}: {e}")

    # 3. Junta tudo com Pydub, aplica crossfade e filtros
    print("Juntando áudios e aplicando filtros de estúdio...")
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
            print(f"Erro ao unir arquivo {arquivo}: {e}")

    # Seu tratamento de áudio original
    audio_final = audio_final.high_pass_filter(180)
    audio_final = normalize(audio_final, headroom=3.0)

    # 4. Salva o resultado
    caminho_saida_wav = os.path.join(temp_dir, "genvox_saida_tratada.wav")
    audio_final.export(caminho_saida_wav, format="wav")

    # 5. Limpa os pedaços temporários
    for arquivo in arquivos_de_audio_gerados:
        try: os.remove(arquivo)
        except: pass

    print("Áudio finalizado com perfeição!")
    return caminho_saida_wav

# =======================================================================
# CONFIGURAÇÃO DO FRONT-END (INTERFACE GRADIO)
# =======================================================================

def atualizar_interface(modo):
    if modo == "Clonar Voz":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

with gr.Blocks(title="GenVox", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎙️ GenVox - Studio")
    gr.Markdown("Bem-vindo ao GenVox. A inteligência artificial foi ajustada para locução natural.")

    with gr.Row():
        with gr.Column(scale=1):
            radio_modo = gr.Radio(choices=["Clonar Voz", "Usar Voz Padrão"], value="Clonar Voz", label="1. Modo de Geração")
            input_audio_clone = gr.Audio(type="filepath", label="Upload de Áudio de Referência (Clonagem)", visible=True)
            dropdown_voz_padrao = gr.Dropdown(choices=vozes_padrao, label="Selecione a Voz Padrão", visible=False)
            
            radio_modo.change(fn=atualizar_interface, inputs=radio_modo, outputs=[input_audio_clone, dropdown_voz_padrao])

            input_texto = gr.Textbox(label="2. Roteiro", placeholder="Escreva o texto...", lines=5)

            with gr.Accordion("3. Ajustes Finos", open=False):
                slider_temp = gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.01, label="Temperatura")
                slider_vel = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.01, label="Velocidade")
                slider_topp = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.01, label="Top-P")

            btn_gerar = gr.Button("🚀 GERAR ÁUDIO", variant="primary")

        with gr.Column(scale=1):
            saida_audio = gr.Audio(label="Resultado da Geração", interactive=False)

    btn_gerar.click(
        fn=gerar_audio,
        inputs=[radio_modo, input_audio_clone, dropdown_voz_padrao, input_texto, slider_temp, slider_vel, slider_topp],
        outputs=saida_audio
    )

if __name__ == "__main__":
    interface.launch(share=True, debug=True)
