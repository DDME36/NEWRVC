import sys
import os
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
from tabs.train.train import train_tab
from tabs.download.download import download_tab
from tabs.settings.settings import settings_tab

# Prerequisites
run_prerequisites_script(pretraineds_hifigan=True, models=True, exe=True, smartcutter=True)

import assets.themes.loadThemes as loadThemes
theme = loadThemes.load_theme() or "ParityError/Interstellar"

# Workflow imports
from workflow.youtube_dl import get_youtube_audio
from workflow.audio_separator import separate_audio

# ─── Helpers ─────────────────────────────────────────────────────────
model_root = os.path.join(core_dir, "logs")

def get_models():
    if not os.path.exists(model_root):
        return []
    return sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(model_root)
        for f in files
        if f.endswith((".pth", ".uvmp"))
        and not (f.startswith("G_") or f.startswith("D_"))
    ])

def get_indexes():
    if not os.path.exists(model_root):
        return []
    return sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(model_root)
        for f in files
        if f.endswith(".index") and "trained" not in f
    ])

def match_index(model_path):
    if not model_path:
        return ""
    model_dir = os.path.dirname(model_path)
    model_base = os.path.splitext(os.path.basename(model_path))[0].split("_")[0]
    try:
        for f in os.listdir(model_dir):
            if f.endswith(".index") and model_base.lower() in f.lower():
                return os.path.join(model_dir, f)
    except Exception:
        pass
    return ""

def mix_audio(vocals_path, instrumental_path, vocals_volume=1.0, instrumental_volume=1.0):
    """Mix converted vocals back with instrumental."""
    voc_data, voc_sr = sf.read(vocals_path)
    inst_data, inst_sr = sf.read(instrumental_path)
    
    # Resample if needed (match to vocal sample rate)
    if inst_sr != voc_sr:
        import librosa
        inst_data = librosa.resample(inst_data.T if inst_data.ndim > 1 else inst_data, 
                                      orig_sr=inst_sr, target_sr=voc_sr)
        if inst_data.ndim > 1:
            inst_data = inst_data.T
    
    # Match lengths
    min_len = min(len(voc_data), len(inst_data))
    voc_data = voc_data[:min_len]
    inst_data = inst_data[:min_len]
    
    # Ensure both are same shape (mono/stereo)
    if voc_data.ndim == 1 and inst_data.ndim > 1:
        voc_data = np.stack([voc_data, voc_data], axis=-1)
    elif voc_data.ndim > 1 and inst_data.ndim == 1:
        inst_data = np.stack([inst_data, inst_data], axis=-1)
    
    mixed = (voc_data * vocals_volume) + (inst_data * instrumental_volume)
    # Prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    
    out_path = os.path.join(root_dir, "workflow_output", "covers", "cover_output.wav")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, mixed, voc_sr)
    return out_path


# ─── Tab 1: AI Song Cover ───────────────────────────────────────────
def build_song_cover_tab():
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
                                info="Protects voiceless consonants. Higher = more protection.")
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

    # ─── Event Handlers ──────────────────────────────────────────────
    def download_yt(url):
        if not url: return gr.update(), gr.update()
        try:
            path = get_youtube_audio(url, os.path.join(root_dir, "workflow_output", "downloads"))
            return path, path
        except Exception as e:
            gr.Warning(f"Download error: {e}")
            return gr.update(), gr.update()

    def use_upload(audio_path):
        return audio_path

    def do_separate(audio_path, sep_choice):
        if not audio_path:
            gr.Warning("No audio to separate! Download or upload first.")
            return None, None
        use_demucs = "Demucs" in sep_choice
        try:
            v, i = separate_audio(
                input_audio_path=audio_path,
                output_dir=os.path.join(root_dir, "workflow_output", "separated"),
                use_demucs=use_demucs,
            )
            return v, i
        except Exception as e:
            gr.Warning(f"Separation error: {e}")
            return None, None

    def do_convert(vocals_path, model, index, p, f0, ir, fr, rmr, prot, emb, fmt, split, at, clean):
        if not vocals_path:
            gr.Warning("No vocals to convert! Run separation first.")
            return None
        if not model:
            gr.Warning("No voice model selected!")
            return None
        out_path = os.path.join(root_dir, "workflow_output", "converted", "converted_vocals.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            msg, result_path = run_infer_script(
                pitch=p, filter_radius=fr, index_rate=ir,
                volume_envelope=rmr, protect=prot, f0_method=f0,
                input_path=vocals_path, output_path=out_path,
                pth_path=model, index_path=index or "",
                split_audio=split, f0_autotune=at, f0_autotune_strength=1.0,
                clean_audio=clean, clean_strength=0.3,
                export_format=fmt, f0_file=None,
                embedder_model=emb,
            )
            return result_path
        except Exception as e:
            gr.Warning(f"Conversion error: {e}")
            return None

    def do_mix(converted, instrumental, vvol, ivol):
        if not converted or not instrumental:
            gr.Warning("Need both converted vocals and instrumental to mix!")
            return None
        try:
            return mix_audio(converted, instrumental, vvol, ivol)
        except Exception as e:
            gr.Warning(f"Mix error: {e}")
            return None

    def refresh_models():
        m = get_models()
        i = get_indexes()
        return gr.update(choices=m), gr.update(choices=i)

    # Wire events
    dl_btn.click(download_yt, [yt_url], [input_audio, upload_audio])
    upload_audio.change(use_upload, [upload_audio], [input_audio])
    sep_btn.click(do_separate, [input_audio, sep_model], [vocals_preview, inst_preview])
    refresh_btn.click(refresh_models, [], [model_file, index_file])
    model_file.change(lambda m: match_index(m), [model_file], [index_file])
    
    convert_btn.click(
        do_convert,
        [vocals_preview, model_file, index_file, pitch, f0_method,
         index_rate, filter_radius, rms_mix_rate, protect,
         embedder_model, export_format, split_audio, autotune, clean_audio],
        [converted_audio],
    )
    mix_btn.click(do_mix, [converted_audio, inst_preview, vocal_vol, inst_vol], [final_cover])


# ═══════════════════════════════════════════════════════════════════
#  BUILD THE APP
# ═══════════════════════════════════════════════════════════════════
with gr.Blocks(title="NEWRVC 🚀", theme=theme) as app:
    gr.Markdown(
        "# 🚀 NEWRVC\n"
        "**High-Performance Voice Conversion** — RingFormer · PCPH-GAN · Spin Embedder\n\n"
        "Built on [codename-rvc-fork-4](https://github.com/codename0og/codename-rvc-fork-4) core engine"
    )

    with gr.Tab("🎵 AI Song Cover"):
        build_song_cover_tab()

    with gr.Tab("🏋️ Train Voice"):
        train_tab()

    with gr.Tab("⬇️ Download Models"):
        download_tab()

    with gr.Tab("⚙️ Settings"):
        settings_tab()


if __name__ == "__main__":
    should_share = "--share" in sys.argv
    app.launch(server_port=7897, inbrowser=True, share=should_share)
