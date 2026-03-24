# NEWRVC

NEWRVC is a high-performance, next-generation Retrieval-based Voice Conversion (RVC) environment. It merges the powerful architecture of **codename-rvc-fork-4** with the user-friendly workflow and automation features inspired by **Ultimate RVC**.

## Features
- **Advanced Vocoders & Embedders**: Native support for RingFormer, PCPH-GAN, and Spin embedders for extreme vocal clarity and performance.
- **Auto YouTube Download**: Download and process audio directly from YouTube via the UI using `yt-dlp`.
- **Integrated Vocal Separation**: Instantly separate vocals and instrumentals without needing external heavy tools like UVR5, powered by `python-audio-separator` (MDX Kim Vocals 2 and Demucs).
- **Gradio 5 UI**: A fast, responsive, and modern interface layout combining Inference, Training, and Workflows into a single smooth experience.

## Installation (Local)
1. Ensure you have Python 3.11 installed.
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Gradio Web UI:
   ```bash
   python app.py
   ```

## Google Colab
You can easily run this project on Google Colab by opening the included `NewRVC_Colab.ipynb` file.

## Credits & Acknowledgements 
This project wouldn't be possible without the incredible work from the open-source community:
- **[codename0og / codename-rvc-fork-4](https://github.com/codename0og/codename-rvc-fork-4)**: For the high-performance core engine, advanced vocoder integrations, and core RVC UI logic.
- **[JackismyShephard / ultimate-rvc](https://github.com/JackismyShephard/ultimate-rvc)**: For the inspiration behind the automated workflows, caching systems, and one-click separation logic.
- **[Applio](https://github.com/IAHispano/Applio)**: The foundational base upon which the codename fork was built.
