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


def _mono_numpy(waveform: torch.Tensor) -> np.ndarray:
    """[B, C, T] or [C, T] -> mono (T,) numpy array."""
    if waveform.dim() == 3:
        waveform = waveform[0]
    return waveform.detach().cpu().float().mean(dim=0).numpy()


def _rms_envelope(mono: np.ndarray, sr: int, hop_ms: float = 10.0):
    hop = max(1, int(sr * hop_ms / 1000.0))
    n_hops = max(1, int(np.ceil(len(mono) / hop)))
    env = np.zeros(n_hops, dtype=np.float64)
    for i in range(n_hops):
        seg = mono[i * hop: i * hop + hop]
        if len(seg) == 0:
            env[i] = env[i - 1] if i > 0 else 0.0
            continue
        env[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    return env, hop


def _smooth_envelope_follower(target_db: np.ndarray, sr: int, hop: int,
                               attack_ms: float, release_ms: float) -> np.ndarray:
    """Exponential attack/release smoothing, starting from 0 (no reduction)."""
    hop_rate = sr / hop
    attack_coeff = np.exp(-1.0 / max(1e-6, hop_rate * (attack_ms / 1000.0)))
    release_coeff = np.exp(-1.0 / max(1e-6, hop_rate * (release_ms / 1000.0)))
    smoothed = np.zeros_like(target_db)
    prev = 0.0
    for i, target in enumerate(target_db):
        coeff = attack_coeff if target > prev else release_coeff
        prev = coeff * prev + (1.0 - coeff) * target
        smoothed[i] = prev
    return smoothed


class SidechainDuck:
    """
    Reduces `target_audio` (e.g. SFX) only during moments where
    `reference_audio` (e.g. voice) is actually loud/active, based on a
    time-varying envelope — not a single flat gain applied across the whole
    clip. Where the reference is quiet or silent, the target is left at full
    level; only the portions overlapping reference activity get ducked, up
    to `max_duck_db`. Never boosts, only ever reduces.
    """

    CATEGORY = "audio/loudness"
    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "avg_duck_db")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_audio": ("AUDIO",),
                "target_audio": ("AUDIO",),
                "threshold_db": (
                    "FLOAT",
                    {"default": -35.0, "min": -80.0, "max": 0.0, "step": 1.0},
                ),
                "max_duck_db": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.0, "max": 40.0, "step": 0.5},
                ),
                "attack_ms": (
                    "FLOAT",
                    {"default": 15.0, "min": 1.0, "max": 500.0, "step": 1.0},
                ),
                "release_ms": (
                    "FLOAT",
                    {"default": 250.0, "min": 1.0, "max": 2000.0, "step": 5.0},
                ),
            }
        }

    def process(self, reference_audio, target_audio, threshold_db=-35.0,
                max_duck_db=12.0, attack_ms=15.0, release_ms=250.0):
        ref_mono = _mono_numpy(reference_audio["waveform"])
        ref_sr = reference_audio["sample_rate"]

        target_waveform = target_audio["waveform"]
        if target_waveform.dim() == 2:
            target_waveform = target_waveform.unsqueeze(0)
        target_sr = target_audio["sample_rate"]
        target_n = target_waveform.shape[-1]

        env, hop = _rms_envelope(ref_mono, ref_sr)
        env_db = 20.0 * np.log10(env + 1e-9)
        duck_db_raw = np.clip(env_db - threshold_db, 0.0, max_duck_db)
        duck_db_smoothed = _smooth_envelope_follower(
            duck_db_raw, ref_sr, hop, attack_ms, release_ms
        )

        hop_times = np.arange(len(duck_db_smoothed)) * hop / float(ref_sr)
        target_times = np.arange(target_n) / float(target_sr)
        duck_db_per_sample = np.interp(
            target_times, hop_times, duck_db_smoothed,
            left=duck_db_smoothed[0] if len(duck_db_smoothed) else 0.0,
            right=0.0,
        )

        gain_per_sample = 10.0 ** (-duck_db_per_sample / 20.0)
        gain_tensor = torch.from_numpy(gain_per_sample.astype(np.float32))
        gain_tensor = gain_tensor.to(target_waveform.device).view(1, 1, -1)
        out_waveform = target_waveform * gain_tensor

        out_audio = {"waveform": out_waveform, "sample_rate": target_sr}
        return (out_audio, float(np.mean(duck_db_per_sample)))


NODE_CLASS_MAPPINGS = {
    "AsymmetricLoudnessLimiter": AsymmetricLoudnessLimiter,
    "RelativeDuckingGain": RelativeDuckingGain,
    "SidechainDuck": SidechainDuck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsymmetricLoudnessLimiter": "Asymmetric Loudness Limiter",
    "RelativeDuckingGain": "Relative Ducking Gain (vs Reference)",
    "SidechainDuck": "Sidechain Duck (Time-Varying)",
}
