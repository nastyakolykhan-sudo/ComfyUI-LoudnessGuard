import numpy as np
import torch

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None


def _require_pyloudnorm():
    if pyln is None:
        raise ImportError(
            "pyloudnorm is required for ComfyUI-LoudnessGuard. Install it with: "
            "pip install pyloudnorm"
        )


def _to_mono_or_stereo_numpy(waveform: torch.Tensor) -> np.ndarray:
    """ComfyUI AUDIO waveform is [B, C, T]. pyloudnorm wants (T,) or (T, C)."""
    if waveform.dim() == 3:
        waveform = waveform[0]
    arr = waveform.detach().cpu().transpose(0, 1).numpy().astype(np.float32)
    return np.ascontiguousarray(arr)


def _measured_lufs(waveform: torch.Tensor, sample_rate: int) -> float:
    _require_pyloudnorm()
    arr = _to_mono_or_stereo_numpy(waveform)
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(arr)
    if not np.isfinite(loudness):
        # Silence / near-silence / too-short clips can make pyloudnorm return
        # -inf or NaN. Treat as "nothing to correct" rather than propagating
        # a NaN gain into the output (see kijai/ComfyUI-WanVideoWrapper#1985
        # for what happens when this case isn't guarded).
        return float("nan")
    return float(loudness)


def _apply_gain_db(waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
    if gain_db == 0.0:
        return waveform
    gain = 10.0 ** (gain_db / 20.0)
    return waveform * gain


class AsymmetricLoudnessLimiter:
    """
    Reduces audio to a target LUFS ceiling only when it's already louder than
    that target. Audio that's already at or below the target passes through
    completely unchanged (gain_db == 0.0, bit-identical output) — unlike a
    standard LUFS normalizer, this never boosts quiet audio up.
    """

    CATEGORY = "audio/loudness"
    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT")
    RETURN_NAMES = ("audio", "measured_lufs", "gain_applied_db")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_lufs": (
                    "FLOAT",
                    {"default": -20.0, "min": -60.0, "max": 0.0, "step": 0.5},
                ),
            }
        }

    def process(self, audio, target_lufs=-20.0):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        measured = _measured_lufs(waveform, sample_rate)

        if np.isnan(measured) or measured <= target_lufs:
            gain_db = 0.0
            out_waveform = waveform
        else:
            gain_db = target_lufs - measured
            out_waveform = _apply_gain_db(waveform, gain_db)

        out_audio = {"waveform": out_waveform, "sample_rate": sample_rate}
        measured_out = 0.0 if np.isnan(measured) else measured
        return (out_audio, float(measured_out), float(gain_db))


class RelativeDuckingGain:
    """
    Compares an SFX/effects track against a reference (typically dialogue)
    track and computes the gain needed to keep the SFX at least `headroom_db`
    quieter than the reference — only ever a reduction (gain_db <= 0.0),
    never a boost. Feed the `gain_db` output into a mixer node's gain input
    (e.g. right-click a numeric widget on a "Mix Audio Tracks" style node and
    "Convert Widget to Input") rather than applying gain here directly, so
    you can keep using your existing mixer.
    """

    CATEGORY = "audio/loudness"
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("gain_db", "reference_lufs", "target_track_lufs")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_audio": ("AUDIO",),
                "target_audio": ("AUDIO",),
                "headroom_db": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
            }
        }

    def process(self, reference_audio, target_audio, headroom_db=6.0):
        ref_lufs = _measured_lufs(
            reference_audio["waveform"], reference_audio["sample_rate"]
        )
        target_lufs = _measured_lufs(
            target_audio["waveform"], target_audio["sample_rate"]
        )

        if np.isnan(ref_lufs) or np.isnan(target_lufs):
            gain_db = 0.0
        else:
            allowed_max = ref_lufs - headroom_db
            gain_db = min(0.0, allowed_max - target_lufs)

        ref_out = 0.0 if np.isnan(ref_lufs) else ref_lufs
        target_out = 0.0 if np.isnan(target_lufs) else target_lufs
        return (float(gain_db), float(ref_out), float(target_out))


NODE_CLASS_MAPPINGS = {
    "AsymmetricLoudnessLimiter": AsymmetricLoudnessLimiter,
    "RelativeDuckingGain": RelativeDuckingGain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsymmetricLoudnessLimiter": "Asymmetric Loudness Limiter",
    "RelativeDuckingGain": "Relative Ducking Gain (vs Reference)",
}
