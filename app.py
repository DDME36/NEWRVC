import sys
import os
import asyncio
import gradio as gr

# Suppress noisy Windows ProactorEventLoop connection-reset errors.
try:
    from asyncio.proactor_events import _ProactorBasePipeTransport
    _original_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost
    def _patched_call_connection_lost(self, exc):
        try:
            _original_call_connection_lost(self, exc)
        except ConnectionResetError:
            pass
    _ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost
except Exception:
    pass

# Setup paths so core codebase works cleanly
root_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(root_dir, 'core')

sys.path.append(core_dir)
sys.path.append(root_dir)

# Change cwd to core_dir so internal relative paths (e.g., assets/) work correctly for the core app
os.chdir(core_dir)

from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.tts.tts import tts_tab
from tabs.utilities.utilities import utilities_tab
from tabs.download.download import download_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.settings.settings import settings_tab

# Core prerequisites
from core import run_prerequisites_script
run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=True,
    smartcutter=True,
)

import assets.installation_checker as installation_checker
installation_checker.check_installation()
import assets.themes.loadThemes as loadThemes
CodenameViolet = loadThemes.load_theme() or "ParityError/Interstellar"

# Import our workflow tools
from workflow.youtube_dl import get_youtube_audio
from workflow.audio_separator import separate_audio

def song_cover_ui():
    with gr.Column():
        gr.Markdown("## 🎵 Audio Download & Separation (Workflow)")
        gr.Markdown("Download a YouTube song and automatically separate Vocals & Instrumental using **MDX Kim Vocals 2** or **Demucs** without needing the heavy UVR5.")
        
        with gr.Row():
            yt_url = gr.Textbox(label="YouTube URL", placeholder="https://youtube.com/watch?v=...")
            dl_btn = gr.Button("Download Audio", variant="primary")
        
        output_audio = gr.Audio(label="Downloaded / Input Audio", type="filepath")
        
        with gr.Row():
            use_demucs = gr.Checkbox(label="Use Demucs (htdemucs_ft) instead of MDX Kim Vocals 2 (UVR-MDX-NET-Voc_FT)?", value=False)
            sep_btn = gr.Button("Separate Vocals & Instrumental", variant="primary")
            
        with gr.Row():
            vocals_out = gr.Audio(label="Vocals", type="filepath", interactive=False)
            inst_out = gr.Audio(label="Instrumental", type="filepath", interactive=False)
            
    def dl_youtube(url):
        if not url:
            return None
        # Save outside the core directory
        out_path = get_youtube_audio(url, os.path.join(root_dir, "workflow_output", "downloads"))
        return out_path
        
    def do_separate(audio_in, demucs):
        if not audio_in:
            return None, None
        v, i = separate_audio(
            input_audio_path=audio_in, 
            output_dir=os.path.join(root_dir, "workflow_output", "separated"), 
            use_demucs=demucs
        )
        return v, i

    dl_btn.click(dl_youtube, inputs=[yt_url], outputs=[output_audio])
    sep_btn.click(do_separate, inputs=[output_audio, use_demucs], outputs=[vocals_out, inst_out])

with gr.Blocks(title="NEWRVC 🚀", theme=CodenameViolet) as app:
    gr.Markdown("# NEWRVC: High-Performance Engine + Ultimate Workflows 🚀")
    gr.Markdown("Featuring RingFormer / PCPH-GAN Vocoders + Spin Embedders via codename-rvc-fork-4 core, plus integrated workflows.")
    
    with gr.Tab("1. Song Cover & Separation"):
        song_cover_ui()
        
    with gr.Tab("2. RVC Inference"):
        inference_tab()
        
    with gr.Tab("3. RVC Training"):
        train_tab()
        
    with gr.Tab("TTS"):
        tts_tab()

    with gr.Tab("Voice Blender"):
        voice_blender_tab()
        
    with gr.Tab("Download Models"):
        download_tab()

    with gr.Tab("Utilities"):
        utilities_tab()

    with gr.Tab("Settings"):
        settings_tab()

if __name__ == "__main__":
    app.launch(server_port=7897, inbrowser=True, share=False)
