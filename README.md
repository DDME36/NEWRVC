<div align="center">

# New RVC ❤️

**High-Performance AI Voice Conversion — Powered by Ultimate RVC Architecture**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDME36/NEWRVC/blob/main/NewRVC_Colab.ipynb)

</div>

---

An advanced fork of [Ultimate RVC](https://github.com/JackismyShephard/ultimate-rvc) with a custom **Red theme**, optimized pipeline, and enhanced core engine features. New RVC provides a professional, clean Gradio 5 interface for generating AI song covers, speech synthesis, and training custom voice models.

![New RVC Web Interface](images/webui_generate.png?raw=true)

## 🔥 What Makes New RVC Different?

### ⚡ The Absolute Best Core Engine
Underneath the beautiful UI is the absolute beast of an engine:

- Native **RingFormer** support
- Experimental **PCPH-GAN** architecture
- Next-gen **Spin Models** embedder
- `uv` Python package manager integration for Colab builds that take seconds, not minutes

### 🎨 Custom Red Theme
A meticulously crafted `#ef4444` red accent theme with Google's Asap font — professional and premium-feeling.

### 🎵 Full Pipeline
- **One-click generation**: Source → Separate → Convert → Mix in a single button
- **Multi-step generation**: Fine control over every stage
- **Speech synthesis**: TTS with any RVC voice model
- **Voice model training**: Full suite with dataset preprocessing, feature extraction, and model training

## Features (Inherited from Ultimate RVC)

- Easy and automated setup using launcher scripts
- Advanced voice conversion with FCPE, RMVPE, Crepe pitch extraction
- Multiple embedder models including Spin and ContentVec
- Pre/post-processing: autotuning, noise reduction, reverb
- Caching system for faster re-processing
- Multi-step generation tabs for experimentation
- Custom configuration save/load system
- Gradio 5 with Python 3.12+ support

## ☁️ Google Colab

Don't have a strong GPU? Run New RVC directly in the cloud:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDME36/NEWRVC/blob/main/NewRVC_Colab.ipynb)

## 💻 Local Setup

### Prerequisites
- Git
- Windows or Debian-based Linux

### Install

```console
git clone https://github.com/DDME36/NEWRVC.git
cd NEWRVC
./urvc install
```

### Run

```console
./urvc run
```

Once you see `Running on local URL: http://127.0.0.1:7860`, click the link to open the app.

### Update

```console
./urvc update
```

### Development Mode

```console
./urvc dev
```

## Usage

### Download Models

Navigate to `Models` > `Download`, paste the URL to a zip containing `.pth` and `.index` files, give it a unique name, and click **Download**.

### Generate Song Covers

**One-click**: Select source type, paste URL or upload file, choose voice model, click **Generate**.

**Multi-step**: Use the accordion steps to fine-tune each stage individually.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `URVC_MODELS_DIR` | Models storage directory | `./models` |
| `URVC_AUDIO_DIR` | Audio files directory | `./audio` |
| `URVC_TEMP_DIR` | Temporary files directory | `./temp` |
| `YT_COOKIEFILE` | YouTube cookies for download | None |
| `URVC_CONFIG` | Custom config name to load | Default |
| `URVC_ACCELERATOR` | `cuda` or `rocm` | `cuda` |

## 🏆 Credits

- **UI Architecture**: [Ultimate RVC](https://github.com/JackismyShephard/ultimate-rvc) by JackismyShephard
- **Core Engine**: codename-rvc-fork-4
- **Tools**: yt-dlp, audio-separator (MDX / Demucs), Gradio 5

## Terms of Use

The use of converted voice for the following purposes is prohibited:
- Criticizing or attacking individuals
- Political advocacy or opposing specific ideologies
- Selling voice models or generated voice clips
- Impersonation with malicious intent
- Fraudulent purposes leading to identity theft

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.
