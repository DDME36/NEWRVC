import os
from audio_separator.separator import Separator

def separate_audio(
    input_audio_path: str,
    output_dir: str,
    model_name: str = "UVR-MDX-NET-Voc_FT", # Default is MDX Kim Vocals 2
    use_demucs: bool = False,
    demucs_model: str = "htdemucs_ft"
) -> tuple[str, str]:
    """
    Separates the input audio file into vocals (primary) and instrumental (secondary).
    Returns (vocals_path, instrumental_path).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    actual_model = demucs_model if use_demucs else model_name
    
    # Initialize the separator
    separator = Separator(
        output_dir=output_dir,
        output_format="WAV",
        sample_rate=44100,
        mdx_params={
            "hop_length": 1024,
            "segment_size": 256,
            "overlap": 0.25,
            "batch_size": 1,
            "enable_denoise": True,
        }
    )
    
    separator.load_model(actual_model)
    
    # Run separation
    primary_stem, secondary_stem = separator.separate(input_audio_path)
    
    vocals_path = os.path.join(output_dir, primary_stem)
    instrumental_path = os.path.join(output_dir, secondary_stem)
    
    return vocals_path, instrumental_path
