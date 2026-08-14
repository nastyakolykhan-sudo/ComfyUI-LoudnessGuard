# ComfyUI-LoudnessGuard

Three loudness-correction nodes that only ever reduce level, never boost it — the piece missing from the usual ComfyUI audio toolchain, where every available loudness normalizer (`Music_LufsNormalizer`, `Egregora Audio Gain Match`, WanVideoWrapper's `NormalizeAudioLoudness`) scales symmetrically to a fixed target and will raise quiet clips up along with lowering loud ones.

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

### Sidechain Duck (Time-Varying)

`Relative Ducking Gain` computes a single flat gain for an entire clip based on its overall loudness. That's fine when two tracks don't overlap much in time, but if `target_audio` only partially overlaps with moments where `reference_audio` is active, a flat reduction leaves it quieter than it needs to be during the gaps. This node ducks `target_audio` dynamically instead: it tracks `reference_audio`'s RMS envelope over time, and only pulls `target_audio` down during the stretches where the reference is actually loud — full level elsewhere.

| Input | Type | Default | Notes |
|---|---|---|---|
| `reference_audio` | AUDIO | — | e.g. your dialogue/voice track |
| `target_audio` | AUDIO | — | e.g. your SFX track |
| `threshold_db` | FLOAT | -35.0 | RMS level in the reference above which ducking engages |
| `max_duck_db` | FLOAT | 12.0 | Ceiling on how much reduction is ever applied |
| `attack_ms` | FLOAT | 15.0 | How fast ducking engages when the reference gets loud |
| `release_ms` | FLOAT | 250.0 | How fast it releases back to full level after |

Outputs: `audio` (the ducked `target_audio`), `avg_duck_db` (mean reduction applied across the whole clip, for logging).

`reference_audio` and `target_audio` are assumed to start at the same point in time (e.g. both begin at the start of the same shot/clip) — the reference's envelope is time-mapped onto the target by elapsed seconds, not by sample count, so differing sample rates are handled automatically, but a timing offset between the two isn't.

## Bundling a fixed pipeline into one reusable step

These nodes are single-purpose on purpose — they don't replicate denoise/EQ/compression that already exists and works in packs like `comfyui_audiotools` or `ComfyUI_MusicTools`. If you've settled on a fixed chain of nodes (from this pack and others) that you want to stop re-wiring by hand, use ComfyUI's own **Group Node** feature: select the finished chain, right-click → "Convert to Group Node" (or save it as a subgraph template in newer ComfyUI versions). That gives you a single reusable block with your exact settings baked in, without this repo needing to reimplement anyone else's DSP.

## Why not just use a LUFS normalizer + skip it conditionally?

There's no clean way to do that with plain graph wiring in ComfyUI without a conditional/switch node reading a measured value and branching — these nodes just fold that logic into a single step instead.

## Notes

- Silence, near-silence, or very short clips can make `pyloudnorm` return a non-finite loudness value. Both nodes detect this (`NaN`/`inf` check) and treat it as "nothing to correct" rather than propagating a broken gain value into the output — see [kijai/ComfyUI-WanVideoWrapper#1985](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1985) for what happens when that case isn't guarded.
- Assumes the standard ComfyUI `AUDIO` dict shape: `{"waveform": torch.Tensor[B, C, T], "sample_rate": int}`.
