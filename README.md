# ComfyUI-LoudnessGuard

Two small LUFS-based loudness nodes that only ever reduce level, never boost it — the piece missing from the usual ComfyUI audio toolchain, where every available loudness normalizer (`Music_LufsNormalizer`, `Egregora Audio Gain Match`, WanVideoWrapper's `NormalizeAudioLoudness`) scales symmetrically to a fixed target and will raise quiet clips up along with lowering loud ones.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo>
pip install -r ComfyUI-LoudnessGuard/requirements.txt
```

Restart ComfyUI.

## Nodes

### Asymmetric Loudness Limiter

Measures a clip's integrated LUFS. If it's already at or under `target_lufs`, the audio passes through completely unchanged (`gain_applied_db` = 0.0, bit-identical output). If it's louder than the target, applies exactly the gain needed to bring it down to the target — never more, never a boost.

| Input | Type | Default | Notes |
|---|---|---|---|
| `audio` | AUDIO | — | |
| `target_lufs` | FLOAT | -20.0 | Range -60 to 0 |

Outputs: `audio`, `measured_lufs`, `gain_applied_db`.

### Relative Ducking Gain (vs Reference)

For mixing two tracks (e.g. dialogue + SFX) where the second track should never get louder than the first, but shouldn't be forced up to match it either. Measures both tracks' LUFS and computes the gain needed to keep `target_audio` at least `headroom_db` quieter than `reference_audio` — the result is clamped so it can only ever be zero or negative.

| Input | Type | Default | Notes |
|---|---|---|---|
| `reference_audio` | AUDIO | — | e.g. your dialogue/voice track |
| `target_audio` | AUDIO | — | e.g. your SFX track |
| `headroom_db` | FLOAT | 6.0 | Minimum gap to maintain below the reference |

Outputs: `gain_db`, `reference_lufs`, `target_track_lufs`.

This node doesn't apply the gain itself — feed `gain_db` into a mixer node's gain input (in ComfyUI, right-click a numeric widget → "Convert Widget to Input") so it plugs into whatever mixing node you're already using.

## Why not just use a LUFS normalizer + skip it conditionally?

There's no clean way to do that with plain graph wiring in ComfyUI without a conditional/switch node reading a measured value and branching — these nodes just fold that logic into a single step instead.

## Notes

- Silence, near-silence, or very short clips can make `pyloudnorm` return a non-finite loudness value. Both nodes detect this (`NaN`/`inf` check) and treat it as "nothing to correct" rather than propagating a broken gain value into the output — see [kijai/ComfyUI-WanVideoWrapper#1985](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1985) for what happens when that case isn't guarded.
- Assumes the standard ComfyUI `AUDIO` dict shape: `{"waveform": torch.Tensor[B, C, T], "sample_rate": int}`.
