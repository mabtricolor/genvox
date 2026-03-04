import os
import tempfile
import uuid
import soundfile as sf
import gradio as gr
from pydub import AudioSegment
from pydub.effects import normalize

# 1. Corrige o erro do Matplotlib no Colab
os.environ['MPLBACKEND'] = 'Agg'

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

from f5_tts.api import F5TTS
from huggingface_hub import hf_hub_download

print("Inicializando o ambiente...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo selecionado: {device}")

print("Baixando e carregando o modelo F5-TTS 100% Brasileiro (FirstPixel)...")
print("Isso pode demorar um pouco na primeira execução.")
# O arquivo safetensors fica DENTRO da pasta "pt-br" no repositório
caminho_modelo_br = hf_hub_download(
    repo_id="firstpixel/F5-TTS-pt-br", 
    filename="pt-br/model_last.safetensors"
)

# Inicializa a máquina do F5-TTS usando o cérebro brasileiro
f5tts = F5TTS(model_type="F5-TTS", ckpt_file=caminho_modelo_br)
print("Modelo Brasileiro carregado com sucesso!")

# =======================================================================
# FUNÇÃO DE GERAÇÃO E TRATAMENTO
# =======================================================================
def gerar_audio_f5(arquivo_clone, texto_referencia, texto):
    if not texto or not texto.strip():
        raise gr.Error("O roteiro não pode estar vazio. Por favor, digite o que a voz deve falar.")
    if not arquivo_clone:
        raise gr.Error("Por favor, faça o upload do áudio de referência para clonagem.")
    
    temp_dir = tempfile.gettempdir()
    print("Iniciando a clonagem de voz e processamento do áudio...")
    
    try:
        # A IA transcreve, clona e gera o áudio
        wav, sr, _ = f5tts.infer(
            ref_file=arquivo_clone, 
            ref_text=texto_referencia.strip() if texto_referencia else "", 
            gen_text=texto.strip()
        )
        
        caminho_saida = os.path.join(temp_dir, f"saida_{uuid.uuid4()}.wav")
        sf.write(caminho_saida, wav, sr)
        
        # O seu toque de mágica de Estúdio com Pydub
        audio_final = AudioSegment.from_wav(caminho_saida)
        
        # Escudo contra quebra de tela (áudio vazio)
        if len(audio_final) > 0:
            audio_final = audio_final.high_pass_filter(180)
            audio_final = normalize(audio_final, headroom=3.0)
            audio_final.export(caminho_saida, format="wav")
            print("Geração concluída com sucesso!")
            return caminho_saida
        else:
            raise gr.Error("O áudio gerado está vazio. Tente outro arquivo de referência.")
            
    except Exception as e:
        raise gr.Error(f"Erro durante a geração: {str(e)}")

# =======================================================================
# FRONT-END (INTERFACE GRADIO)
# =======================================================================
with gr.Blocks(title="GenVox - F5 Studio BR", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎙️ GenVox - F5 Studio (Edição Brasil)")
    gr.Markdown("O seu estúdio na nuvem focado em **Clonagem de Voz Ultrarrealista** usando a tecnologia F5-TTS nativa em Português.")

    with gr.Row():
        with gr.Column(scale=1):
            
            gr.Markdown("### 1. A Voz Original (Clone)")
            input_audio_clone = gr.Audio(
                type="filepath", 
                label="Upload do Áudio (O ideal é de 10 a 15 segundos do locutor em Português)"
            )
            input_texto_ref = gr.Textbox(
                label="O que a pessoa diz no áudio acima? (Opcional)", 
                placeholder="Deixe em branco para a IA transcrever sozinha usando o Whisper..."
            )
            
            gr.Markdown("### 2. O Roteiro")
            input_texto = gr.Textbox(
                label="Escreva o que a nova voz deverá falar:", 
                lines=5
            )

            btn_gerar = gr.Button("🚀 CLONAR E GERAR ÁUDIO", variant="primary")

        with gr.Column(scale=1):
            saida_audio = gr.Audio(label="Resultado da Geração", interactive=False)

    btn_gerar.click(
        fn=gerar_audio_f5,
        inputs=[input_audio_clone, input_texto_ref, input_texto],
        outputs=saida_audio
    )

if __name__ == "__main__":
    interface.launch(share=True, debug=True)
