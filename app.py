import sys
import os
import shutil
import zipfile
import tempfile
import gradio as gr
import numpy as np
import soundfile as sf

# Suppress noisy Windows ProactorEventLoop connection-reset errors.
try:
    from asyncio.proactor_events import _ProactorBasePipeTransport
    _original = _ProactorBasePipeTransport._call_connection_lost
    def _patched(self, exc):
        try: _original(self, exc)
        except ConnectionResetError: pass
    _ProactorBasePipeTransport._call_connection_lost = _patched
except Exception:
    pass

# ─── Path Setup ──────────────────────────────────────────────────────
root_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(root_dir, "core")
sys.path.append(core_dir)
sys.path.append(root_dir)
os.chdir(core_dir)

# ─── Core Imports ────────────────────────────────────────────────────
from core import run_infer_script, run_prerequisites_script
from rvc.lib.tools.model_download import download_from_url
from tabs.train.train import train_tab
from tabs.settings.settings import settings_tab

# Import the pretrained downlowder helpers from core
from tabs.download.download import (
    get_pretrained_list,
    get_pretrained_sample_rates,
    download_pretrained_model as core_download_pretrained_model,
)

# Prerequisites
run_prerequisites_script(pretraineds_hifigan=True, models=True, exe=True, smartcutter=True)

# CSS & Theme (Ultimate Red)
theme_path = os.path.join(root_dir, "assets", "themes", "ultimate_red.json")
theme = gr.Theme.load(theme_path) if os.path.exists(theme_path) else "Default"

css = """
h1 { text-align: center; margin-top: 20px; margin-bottom: 20px; }
#generate-tab-button { font-weight: bold !important;}
#manage-tab-button { font-weight: bold !important;}
#train-tab-button { font-weight: bold !important;}
#settings-tab-button { font-weight: bold !important;}
"""

# Workflow imports
from workflow.youtube_dl import get_youtube_audio
from workflow.audio_separator import separate_audio

# ─── Helpers ─────────────────────────────────────────────────────────
model_root = os.path.join(core_dir, "logs")

def get_models():
    if not os.path.exists(model_root): return []
    return sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(model_root)
        for f in files
        if f.endswith((".pth", ".uvmp")) and not (f.startswith("G_") or f.startswith("D_"))
    ])

def get_indexes():
    if not os.path.exists(model_root): return []
    return sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(model_root)
        for f in files
        if f.endswith(".index") and "trained" not in f
    ])

def match_index(model_path):
    if not model_path: return ""
    model_dir = os.path.dirname(model_path)
    model_base = os.path.splitext(os.path.basename(model_path))[0].split("_")[0]
    try:
        for f in os.listdir(model_dir):
            if f.endswith(".index") and model_base.lower() in f.lower():
                return os.path.join(model_dir, f)
    except Exception: pass
    return ""

def refresh_models():
    return gr.update(choices=get_models()), gr.update(choices=get_indexes())

def mix_audio(vocals_path, instrumental_path, vocals_volume=1.0, instrumental_volume=1.0):
    voc_data, voc_sr = sf.read(vocals_path)
    inst_data, inst_sr = sf.read(instrumental_path)
    
    if inst_sr != voc_sr:
        import librosa
        inst_data = librosa.resample(inst_data.T if inst_data.ndim > 1 else inst_data, 
                                      orig_sr=inst_sr, target_sr=voc_sr)
        if inst_data.ndim > 1: inst_data = inst_data.T
    
    min_len = min(len(voc_data), len(inst_data))
    voc_data = voc_data[:min_len]
    inst_data = inst_data[:min_len]
    
    if voc_data.ndim == 1 and inst_data.ndim > 1:
        voc_data = np.stack([voc_data, voc_data], axis=-1)
    elif voc_data.ndim > 1 and inst_data.ndim == 1:
        inst_data = np.stack([inst_data, inst_data], axis=-1)
    
    mixed = (voc_data * vocals_volume) + (inst_data * instrumental_volume)
    peak = np.max(np.abs(mixed))
    if peak > 1.0: mixed = mixed / peak
    
    out_path = os.path.join(root_dir, "workflow_output", "covers", "cover_output.wav")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, mixed, voc_sr)
    return out_path

# ─── Custom Downloader ──────────────────────────────────────────────
def download_custom_voice_model(url, model_name):
    if not url or not model_name:
        raise gr.Error("Please provide both a URL and a Model Name.")
    
    model_name = "".join(c for c in model_name if c.isalnum() or c in (" ", "-", "_")).strip()
    if not model_name:
        raise gr.Error("Invalid model name.")
        
    zips_folder = os.path.join(model_root, "zips")
    os.makedirs(zips_folder, exist_ok=True)
    
    # 1. Clear existing zips to avoid confusion
    for f in os.listdir(zips_folder):
        os.remove(os.path.join(zips_folder, f))
        
    # 2. Download using core logic
    gr.Info(f"Downloading model from {url}...")
    res = download_from_url(url)
    if res != "downloaded":
        raise gr.Error("Failed to download the model from the provided URL.")
    
    # 3. Find the downloaded zip
    downloaded_zip = None
    for f in os.listdir(zips_folder):
        if f.endswith(".zip"):
            downloaded_zip = os.path.join(zips_folder, f)
            break
            
    if not downloaded_zip:
        raise gr.Error("No ZIP file was found after download. Make sure the URL points to a valid zip.")
        
    # 4. Extract into a temp folder
    temp_extract = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
    except Exception as e:
        shutil.rmtree(temp_extract)
        raise gr.Error(f"Failed to extract the downloaded zip file: {e}")
        
    # 5. Search for .pth and .index
    pth_file = None
    index_file = None
    
    for root_dir_walk, _, files in os.walk(temp_extract):
        for f in files:
            if f.endswith(".pth") and "G_" not in f and "D_" not in f:
                pth_file = os.path.join(root_dir_walk, f)
            elif f.endswith(".index") and "trained" not in f:
                index_file = os.path.join(root_dir_walk, f)
                
    if not pth_file:
        shutil.rmtree(temp_extract)
        raise gr.Error("No valid .pth model was found in the downloaded archive.")

    # 6. Move and rename to logs/model_name/
    final_folder = os.path.join(model_root, model_name)
    os.makedirs(final_folder, exist_ok=True)
    
    final_pth = os.path.join(final_folder, f"{model_name}.pth")
    shutil.move(pth_file, final_pth)
    
    if index_file:
        final_index = os.path.join(final_folder, f"{model_name}.index")
        shutil.move(index_file, final_index)
        
    # Cleanup
    shutil.rmtree(temp_extract)
    os.remove(downloaded_zip)
    
    return f"[+] Successfully downloaded and set up Voice Model: {model_name}!"

def update_pretrained_dropdown(model):
    choices = get_pretrained_sample_rates(model)
    return gr.update(choices=choices, value=choices[0] if choices else None)

def download_custom_pretrained(model, sr):
    core_download_pretrained_model(model, sr)
    return f"[+] Successfully downloaded pretrained model {model} ({sr})!"


# ─── TABS BUILDERS ──────────────────────────────────────────────────
def render_generate_tab():
    with gr.Tab("Song covers"):
        models = get_models()
        indexes = get_indexes()
        default_model = models[0] if models else None

        gr.Markdown("### Step 1: Input Audio")
        with gr.Row():
            with gr.Column(scale=3):
                yt_url = gr.Textbox(label="YouTube URL", placeholder="https://youtube.com/watch?v=...")
            with gr.Column(scale=1):
                dl_btn = gr.Button("⬇️ Download", variant="primary", size="lg")
        upload_audio = gr.Audio(label="Or Upload Audio File", type="filepath")
        input_audio = gr.Audio(label="🎧 Input Audio Preview", type="filepath", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### Step 2: Separate Vocals & Instrumental")
        with gr.Row():
            sep_model = gr.Radio(
                choices=["MDX Kim Vocals 2", "Demucs (htdemucs_ft)"],
                value="MDX Kim Vocals 2",
                label="Separation Model",
            )
            sep_btn = gr.Button("🔀 Separate Audio", variant="primary", size="lg")

        with gr.Row():
            vocals_preview = gr.Audio(label="🎤 Vocals", type="filepath", interactive=False)
            inst_preview = gr.Audio(label="🎵 Instrumental", type="filepath", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### Step 3: Voice Conversion (RVC)")
        with gr.Row():
            model_file = gr.Dropdown(
                label="Voice Model", choices=models, value=default_model,
                interactive=True, allow_custom_value=True,
            )
            index_file = gr.Dropdown(
                label="Index File", choices=indexes,
                value=match_index(default_model) if default_model else "",
                interactive=True, allow_custom_value=True,
            )
            refresh_btn = gr.Button("🔄", size="sm")

        with gr.Row():
            pitch = gr.Slider(-24, 24, value=0, step=1, label="Pitch (semitones)",
                              info="↑ = higher voice, ↓ = lower voice. +12 for male→female, -12 for female→male")
            f0_method = gr.Dropdown(
                choices=["rmvpe", "crepe", "crepe-tiny", "fcpe", "hybrid[rmvpe+fcpe]"],
                value="rmvpe", label="F0 Method",
                info="rmvpe is fast & accurate. crepe for singing quality. fcpe is newest."
            )

        with gr.Accordion("🔧 Advanced Voice Settings", open=False):
            with gr.Row():
                index_rate = gr.Slider(0, 1, value=0.75, step=0.05, label="Index Rate",
                                       info="Higher = more model character. Lower = more natural.")
                filter_radius = gr.Slider(0, 10, value=3, step=1, label="Filter Radius",
                                          info="Median filter on pitch. ≥3 reduces breathiness.")
                rms_mix_rate = gr.Slider(0, 1, value=0.25, step=0.05, label="Volume Envelope",
                                         info="0 = keep original loudness. 1 = match model loudness.")
                protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label="Protect",
                                    info="Protects voiceless consonants.")
            with gr.Row():
                embedder_model = gr.Dropdown(
                    choices=["contentvec", "contentvec_base", "chinese-hubert-base", "japanese-hubert-base", "spin"],
                    value="contentvec", label="Embedder Model"
                )
                export_format = gr.Radio(choices=["WAV", "MP3", "FLAC"], value="WAV", label="Export Format")
            with gr.Row():
                split_audio = gr.Checkbox(label="Split Audio (for long files)", value=False)
                autotune = gr.Checkbox(label="Autotune", value=False)
                clean_audio = gr.Checkbox(label="Clean Audio (noise reduction)", value=False)

        convert_btn = gr.Button("🎙️ Convert Voice", variant="primary", size="lg")
        converted_audio = gr.Audio(label="🎤 Converted Vocals", type="filepath", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### Step 4: Mix & Download Final Cover")
        with gr.Row():
            vocal_vol = gr.Slider(0, 2, value=1.0, step=0.05, label="Vocals Volume")
            inst_vol = gr.Slider(0, 2, value=1.0, step=0.05, label="Instrumental Volume")
        mix_btn = gr.Button("🎶 Mix Final Cover", variant="primary", size="lg")
        final_cover = gr.Audio(label="🏆 Final AI Cover", type="filepath", interactive=False)

        # Event setup (similar to before but robust)
        def download_yt(url):
            if not url: raise gr.Error("Please enter a YouTube URL.")
            path = get_youtube_audio(url, os.path.join(root_dir, "workflow_output", "downloads"))
            return path, path
            
        def do_separate(audio_path, sep_choice):
            if not audio_path: raise gr.Error("No audio to separate! Download or upload first.")
            v, i = separate_audio(
                input_audio_path=audio_path,
                output_dir=os.path.join(root_dir, "workflow_output", "separated"),
                use_demucs="Demucs" in sep_choice,
            )
            return v, i

        def do_convert(vocal_path, m, idx, p, f0, ir, fr, rmr, prot, emb, fmt, split, at, clean):
            if not vocal_path: raise gr.Error("No vocals to convert!")
            if not m: raise gr.Error("No voice model selected!")
            out_path = os.path.join(root_dir, "workflow_output", "converted", "converted_vocals.wav")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            run_infer_script(
                pitch=p, filter_radius=fr, index_rate=ir,
                volume_envelope=rmr, protect=prot, f0_method=f0,
                input_path=vocal_path, output_path=out_path,
                pth_path=m, index_path=idx or "",
                split_audio=split, f0_autotune=at, f0_autotune_strength=1.0,
                clean_audio=clean, clean_strength=0.3, export_format=fmt,
                f0_file=None, embedder_model=emb,
            )
            return out_path

        def do_mix(conv, inst, vv, iv):
            if not conv or not inst: raise gr.Error("Need both vocals and instrumental!")
            return mix_audio(conv, inst, vv, iv)

        dl_btn.click(download_yt, [yt_url], [input_audio, upload_audio])
        upload_audio.change(lambda x: x, [upload_audio], [input_audio])
        sep_btn.click(do_separate, [input_audio, sep_model], [vocals_preview, inst_preview])
        refresh_btn.click(refresh_models, [], [model_file, index_file])
        model_file.change(lambda m: match_index(m), [model_file], [index_file])
        convert_btn.click(do_convert, [vocals_preview, model_file, index_file, pitch, f0_method, index_rate, filter_radius, rms_mix_rate, protect, embedder_model, export_format, split_audio, autotune, clean_audio], [converted_audio])
        mix_btn.click(do_mix, [converted_audio, inst_preview, vocal_vol, inst_vol], [final_cover])

def render_models_tab():
    with gr.Accordion("Voice models", open=True):
        gr.Markdown(
            "1. Enter the download **URL** for a .zip file (HuggingFace URL, Google Drive URL, etc.)\n"
            "2. Enter a unique **Model name** for the voice.\n"
            "3. Click **Download 🌐** and it will automatically extract and rename the `.pth` and `.index` files cleanly into `logs/`."
        )
        with gr.Row():
            voice_model_url = gr.Textbox(
                label="Model URL",
                info="URL to a zip file containing the model.",
            )
            voice_model_name = gr.Textbox(
                label="Model name",
                info="Name to assign to this model.",
            )
        with gr.Row(equal_height=True):
            download_voice_btn = gr.Button("Download 🌐", variant="primary", scale=19)
            download_voice_msg = gr.Textbox(label="Output message", interactive=False, scale=20)
            
        download_voice_btn.click(
            fn=download_custom_voice_model,
            inputs=[voice_model_url, voice_model_name],
            outputs=[download_voice_msg]
        )
        
    with gr.Accordion("Pretrained models", open=False):
        pretrained_list = get_pretrained_list()
        default_model = "Titan" if "Titan" in pretrained_list else (pretrained_list[0] if pretrained_list else "")
        default_rates = get_pretrained_sample_rates(default_model) if default_model else []
        
        with gr.Row():
            pretrained_model = gr.Dropdown(
                label="Pretrained model",
                info="Select the pretrained model you want to download.",
                choices=pretrained_list,
                value=default_model,
            )
            pretrained_sample_rate = gr.Dropdown(
                label="Sample rate",
                info="Select the sample rate for the pretrained model.",
                choices=default_rates,
                value=default_rates[0] if default_rates else None,
            )
        
            pretrained_model.change(
                fn=update_pretrained_dropdown,
                inputs=[pretrained_model],
                outputs=[pretrained_sample_rate]
            )
            
        with gr.Row(equal_height=True):
            download_pretrained_btn = gr.Button("Download 🌐", variant="primary", scale=19)
            download_pretrained_msg = gr.Textbox(label="Output message", interactive=False, scale=20)
            
        download_pretrained_btn.click(
            fn=download_custom_pretrained,
            inputs=[pretrained_model, pretrained_sample_rate],
            outputs=[download_pretrained_msg]
        )

# ═══════════════════════════════════════════════════════════════════
#  BUILD THE APP
# ═══════════════════════════════════════════════════════════════════
with gr.Blocks(title="NEWRVC 🚀", theme=theme, css=css) as app:
    gr.HTML("<h1>NEWRVC 🚀</h1>")

    with gr.Tab("Generate", elem_id="generate-tab"):
        render_generate_tab()
        
    with gr.Tab("Models", elem_id="manage-tab"):
        render_models_tab()

    with gr.Tab("Train", elem_id="train-tab"):
        train_tab()

    with gr.Tab("Settings", elem_id="settings-tab"):
        settings_tab()


if __name__ == "__main__":
    should_share = "--share" in sys.argv
    app.launch(server_port=7897, inbrowser=True, share=should_share)
