import os
import shutil
import hashlib

def get_file_hash(file_path: str) -> str:
    """Returns the MD5 hash of a file for caching purposes."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def cache_audio(source_path: str, cache_dir: str) -> str:
    """
    Caches an audio file into a structured directory using its hash.
    Returns the path to the cached file.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        
    file_hash = get_file_hash(source_path)
    extension = os.path.splitext(source_path)[1]
    
    cached_path = os.path.join(cache_dir, f"{file_hash}{extension}")
    
    if not os.path.exists(cached_path):
        shutil.copy2(source_path, cached_path)
        
    return cached_path

def get_intermediate_listen_path(step_name: str, session_dir: str, file_name: str) -> str:
    """
    Generates a path for intermediate listen steps (like isolated vocals before RVC).
    """
    step_dir = os.path.join(session_dir, step_name)
    if not os.path.exists(step_dir):
        os.makedirs(step_dir, exist_ok=True)
    return os.path.join(step_dir, file_name)
