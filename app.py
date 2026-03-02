import os

# 1. Corrige o erro do Matplotlib no Colab (ANTES de qualquer outra importação)
os.environ['MPLBACKEND'] = 'Agg'

# 2. Aceita automaticamente os termos de uso do modelo XTTS-v2
os.environ["COQUI_TOS_AGREED"] = "1"

import tempfile
import torch
import gradio as gr
from TTS.api import TTS

print("Inicializando o ambiente...")

# Verifica se a GPU está disponível. No Colab, isso deve ser "cuda".
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo selecionado: {device}")

print("Carregando o modelo XTTS-v2... Isso pode demorar um pouco na primeira execução.")
# Instancia o modelo XTTS-v2 e envia para a GPU
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Extrai dinamicamente a lista de vozes padrão disponíveis dentro do modelo
vozes_padrao = list(tts.synthesizer.tts_model.speaker_manager.name_to_id.keys())
print("Modelo carregado com sucesso!")

def gerar_audio(modo, arquivo_clone, voz_selecionada, texto, temperatura, velocidade, top_p):
    """
    Função principal que o Gradio chamará ao clicar em 'GERAR ÁUDIO'.
    """
    if not texto or not texto.strip():
        raise gr.Error("O texto não pode estar vazio. Por favor, digite um roteiro.")

    # Parâmetros básicos de geração
    params = {
        "text": texto,
        "language": "pt",
        "temperature": float(temperatura),
        "speed": float(velocidade),
        "top_p": float(top_p)
    }

    # Configura os parâmetros dependendo do modo selecionado
    if modo == "Clonar Voz":
        if not arquivo_clone:
            raise gr.Error("Por favor, faça o upload de um arquivo de áudio de referência para clonar a voz.")
        params["speaker_wav"] = arquivo_clone
    else:
        if not voz_selecionada:
            raise gr.Error("Por favor, selecione uma voz padrão do modelo na lista.")
        params["speaker"] = voz_selecionada

    # Cria um diretório temporário para salvar o áudio final gerado
    temp_dir = tempfile.gettempdir()
    caminho_saida = os.path.join(temp_dir, "genvox_saida.wav")

    print(f"Iniciando a geração de áudio no modo: {modo}...")
    
    # Chama o motor para gerar o arquivo de áudio
    tts.tts_to_file(file_path=caminho_saida, **params)
    
    print("Geração concluída!")
    return caminho_saida

# =======================================================================
# CONFIGURAÇÃO DO FRONT-END (INTERFACE GRADIO)
# =======================================================================

def atualizar_interface(modo):
    """
    Controla a visibilidade: Se for "Clonar Voz", mostra o upload de áudio e esconde a lista.
    Se for "Usar Voz Padrão", esconde o upload e mostra a lista.
    """
    if modo == "Clonar Voz":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

# Cria a estrutura visual da nossa aplicação
with gr.Blocks(title="GenVox", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎙️ GenVox - Studio")
    gr.Markdown("Bem-vindo ao GenVox. Escolha o modo de operação, configure a voz, insira seu texto e gere o áudio!")

    with gr.Row():
        # Coluna da Esquerda (Configurações e Entrada)
        with gr.Column(scale=1):
            
            # 1. Modo de Operação
            radio_modo = gr.Radio(
                choices=["Clonar Voz", "Usar Voz Padrão"], 
                value="Clonar Voz", 
                label="1. Modo de Geração"
            )

            # Elementos Dinâmicos de Voz
            input_audio_clone = gr.Audio(
                type="filepath", 
                label="Upload de Áudio de Referência (Clonagem)",
                visible=True
            )
            dropdown_voz_padrao = gr.Dropdown(
                choices=vozes_padrao, 
                label="Selecione a Voz Padrão", 
                visible=False
            )

            # Liga o Radio Button à função que atualiza a visibilidade da tela
            radio_modo.change(
                fn=atualizar_interface, 
                inputs=radio_modo, 
                outputs=[input_audio_clone, dropdown_voz_padrao]
            )

            # 2. Texto
            input_texto = gr.Textbox(
                label="2. Roteiro", 
                placeholder="Escreva o que a voz deverá falar aqui...", 
                lines=5
            )

            # 3. Ajustes Finos
            with gr.Accordion("3. Ajustes Finos", open=False):
                slider_temp = gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.01, label="Temperatura (Aleatoriedade)")
                slider_vel = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.01, label="Velocidade")
                slider_topp = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.01, label="Top-P (Fidelidade)")

            # Botão de Gerar
            btn_gerar = gr.Button("🚀 GERAR ÁUDIO", variant="primary")

        # Coluna da Direita (Saída de Áudio)
        with gr.Column(scale=1):
            saida_audio = gr.Audio(label="Resultado da Geração", interactive=False)

    # Conecta o clique do botão à função principal
    btn_gerar.click(
        fn=gerar_audio,
        inputs=[
            radio_modo, 
            input_audio_clone, 
            dropdown_voz_padrao, 
            input_texto, 
            slider_temp, 
            slider_vel, 
            slider_topp
        ],
        outputs=saida_audio
    )

if __name__ == "__main__":
    # Inicia a aplicação com share=True para criar um link público acessível pela web
    interface.launch(share=True, debug=True)
