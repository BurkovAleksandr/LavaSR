import torch
import torchaudio
from huggingface_hub import snapshot_download

from LavaSR.enhancer.enhancer import LavaBWE
from LavaSR.denoiser.denoiser import LavaDenoiser
from LavaSR.utils import wav_to_1s_batches, load_wav
from LavaSR.enhancer.linkwitz_merge import FastLRMerge


import os
from huggingface_hub import snapshot_download

HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"


def resolve_snapshot_path(model_id: str) -> str:
    if "/" not in model_id:
        raise ValueError(f"model_id '{model_id}' must be in 'org/name' format")

    org, name = model_id.split("/", 1)

    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    # --- основной путь через refs/main ---
    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()

        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    # --- fallback: берем самый свежий snapshot ---
    if os.path.isdir(snapshots_dir):
        versions = [
            d
            for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]

        if versions:
            versions.sort(reverse=True)  # <-- ВАЖНО: самый новый
            return os.path.join(snapshots_dir, versions[0])

    raise RuntimeError(f"Cached model not found: {model_id}")


def resolve_model_path(model_id: str) -> str:
    """
    Универсальный резолвер:
    1. Явный локальный путь
    2. ENV MODEL_DIR
    3. RunPod HF cache (/runpod-volume)
    4. HF cache (local_files_only)
    5. Скачать модель
    """

    # --- 1. если это уже путь ---
    if os.path.isdir(model_id):
        return model_id

    # --- 2. ENV override ---
    env_path = os.getenv("MODEL_DIR")
    if env_path and os.path.isdir(env_path):
        return env_path

    # --- 3. RunPod cache ---
    try:
        path = resolve_snapshot_path(model_id)
        if os.path.isdir(path):
            return path
    except Exception as exc:
        print(exc)

    # --- 4. HF cache без сети ---
    try:
        return snapshot_download(model_id, local_files_only=True)
    except Exception as exc:
        print(exc)

    # --- 5. fallback: скачать ---
    return snapshot_download(model_id)


class LavaEnhance:
    def __init__(self, model_path="YatharthS/LavaSR", device="cpu"):

        model_path = resolve_model_path(model_path)

        self.device = device
        self.bwe_model = LavaBWE(
            f"{model_path}/enhancer", device=device
        )  ## proposed work
        self.denoiser_model = LavaDenoiser(
            f"{model_path}/denoiser/denoiser.bin", device=device
        )  ## based on UL-UNAS

    def enhance(self, wav, enhance=True, denoise=True, batch=False):
        pad_size = 0
        low_quality_audio = wav

        if batch:
            wav, pad_size = wav_to_1s_batches(wav, 16000)

        if denoise:
            with torch.inference_mode():
                wav = self.denoiser_model.infer(wav)
                wav = torchaudio.functional.resample(wav, 16000, 48000)
        else:
            wav = torchaudio.functional.resample(wav, 16000, 48000)

        if enhance:
            with torch.no_grad():
                wav = self.bwe_model.infer(wav).reshape(-1)
        else:
            wav = wav.reshape(-1)

        return wav

    def load_audio(self, file_path, input_sr=16000, duration=10000, cutoff=None):
        x = load_wav(file_path, resample_to=input_sr, duration=duration).to(self.device)

        if cutoff == None:
            cutoff = input_sr // 2

        self.bwe_model.lr_refiner = FastLRMerge(
            device=self.device, cutoff=cutoff, transition_bins=1024
        )

        return x, input_sr


class LavaEnhance2(LavaEnhance):
    def __init__(self, model_path="YatharthS/LavaSR", device="cpu"):

        model_path = resolve_model_path(model_path)

        self.device = device
        self.bwe_model = LavaBWE(f"{model_path}/enhancer_v2", device=device)
        self.denoiser_model = LavaDenoiser(
            f"{model_path}/denoiser/denoiser.bin", device=device
        )
