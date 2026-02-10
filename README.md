# ucc-librispeech-longform-asr

Utterance-Concatenation Curriculum (UCC) for long-form robustness in CTC ASR by fine-tuning `facebook/wav2vec2-base-960h` and training on synthetic concatenations of utterances (optionally padded with boundary silence).

This repo is fully runnable end-to-end:
- `make all` (default) runs a tiny synthetic "smoke" dataset and produces `artifacts/results.json` + `artifacts/report.md`.
- Full LibriSpeech support is implemented (download, MD5 verification, manifest creation, training configs), but the checked-in `artifacts/*` results are from the smoke run (details below).

## Problem Statement

Standard LibriSpeech fine-tuning is per-utterance (`N=1`) but real deployments see longer audio where utterance boundaries are imperfect (segmentation errors, pauses, concatenations). This project tests a simple curriculum:

- Train on synthetic examples formed by randomly concatenating `N` utterances (UCC), with a schedule that can ramp `N` from 1 to `target_n`.
- Add an ablation on the boundary condition: insert `boundary_silence_sec` between utterances vs no silence.
- Evaluate both standard per-utterance WER/CER and a synthetic long-form evaluation that stresses boundary seams.

## Dataset Provenance

1) **Smoke dataset (default, generated locally)**  
`data.kind: smoke` (see `configs/smoke.yaml`) generates short 1.2s WAV clips with synthetic tones + noise and random transcripts drawn from a fixed vocab (`ucc_asr/data/smoke.py`). This is for pipeline validation, not ASR quality.

2) **LibriSpeech (implemented)**  
`data.kind: librispeech` downloads OpenSLR resource 12 tarballs from:
- `https://www.openslr.org/resources/12/train-clean-100.tar.gz`
- `https://www.openslr.org/resources/12/dev-clean.tar.gz`
- `https://www.openslr.org/resources/12/dev-other.tar.gz`
- `https://www.openslr.org/resources/12/test-clean.tar.gz`
- `https://www.openslr.org/resources/12/test-other.tar.gz`

It also downloads `https://www.openslr.org/resources/12/md5sum.txt` and verifies MD5 when an entry exists (`ucc_asr/io.py`, `ucc_asr/data/prepare.py`), then extracts with a path-traversal guard and builds JSONL manifests by pairing `*.flac` with `*.trans.txt` (`ucc_asr/data/librispeech.py`).

## Methodology (What’s Implemented)

**Model:** Hugging Face `Wav2Vec2ForCTC` + `Wav2Vec2Processor` (`ucc_asr/modeling.py`). Training is standard CTC fine-tuning; decoding is greedy argmax at eval (`ucc_asr/eval.py`).

**UCC training data:** `RandomConcatIterableDataset` (`ucc_asr/data/datasets.py`)
- Samples `n` utterances uniformly at random and concatenates waveforms.
- Transcript is the space-joined concatenation of transcripts.
- Optional boundary padding via `concat_with_silence(..., silence_sec=boundary_silence_sec)` (`ucc_asr/audio.py`).
- Curriculum schedule:
  - `fixed`: always use `target_n`
  - `ramp`: linearly ramp `n` from 1 to `target_n` over `rampup_fraction * max_steps`

**Long-form synthetic eval:** `make_longform_examples(k=...)` (`ucc_asr/longform.py`) concatenates `k` utterances from the test split into longer clips; evaluation reports:
- long-form `WER`, `CER` (via `jiwer` on normalized text)
- `RTF` (wall-time / audio duration)
- seam-local error rate: edits within `+/- seam_window_words` of each seam, computed by explicit Levenshtein alignment (`ucc_asr/metrics.py`)

## Baselines and Ablations (As Configured)

The default smoke config (`configs/smoke.yaml`) defines three experiments:

| experiment | UCC | target_n | schedule | rampup_fraction | boundary_silence_sec |
| --- | --- | --- | --- | --- | --- |
| `baseline` | off | 1 | n/a | n/a | 0.0 |
| `ucc_n4_sil200_ramp` | on | 4 | ramp | 0.5 | 0.2 |
| `ablation_no_silence` | on | 4 | ramp | 0.5 | 0.0 |

LibriSpeech training configs mirror the same idea with longer runs (e.g., `max_steps: 20000`, AMP on CUDA, gradient checkpointing enabled): `configs/librispeech_baseline.yaml`, `configs/librispeech_ucc_n4_sil200_ramp.yaml`, `configs/librispeech_ablation_no_silence.yaml`.

## Results (Exact Numbers From This Repo)

The checked-in results were generated from `configs/smoke.yaml` and written to `artifacts/results.json` at `2026-02-10T18:08:42Z` (aggregate) and `artifacts/report.md` at `2026-02-10T18:08:46Z`.
There are no figures; the repo produces markdown tables (see `artifacts/report.md`).

**Standard evaluation (WER)**  
Source: `artifacts/report.md` section "Standard Evaluation" (derived from `artifacts/results.json`).

| experiment | dev_wer | test_wer |
| --- | --- | --- |
| baseline | 1.0000 | 1.0000 |
| ucc_n4_sil200_ramp | 1.0000 | 1.0000 |
| ablation_no_silence | 1.0000 | 1.0000 |

**Long-form synthetic evaluation**  
Source: `artifacts/report.md` section "Long-Form Synthetic Evaluation".

| experiment | longform_wer | longform_cer | seam_local_error_rate | rtf |
| --- | --- | --- | --- | --- |
| baseline | 1.0000 | 1.0000 | 1.0000 | 0.0189 |
| ucc_n4_sil200_ramp | 1.0000 | 1.0000 | 1.0000 | 0.0185 |
| ablation_no_silence | 1.0000 | 1.0000 | 1.0000 | 0.0182 |

**Eval/runtime traceability (from `artifacts/results.json`)**
- `device`: `cpu` for all three experiments in this run.
- `checkpoint`: `runs/<experiment>/checkpoints/best.pt` for `baseline`, `ucc_n4_sil200_ramp`, and `ablation_no_silence`.

Interpretation: on the synthetic smoke task (4 optimization steps, random transcripts unrelated to tones), the model does not learn; the tables above are primarily a sanity check that the pipeline runs and produces comparable artifacts across baseline/UCC/ablation.

## Repro Instructions

### Smoke (Matches `artifacts/*`)

```bash
make all
```

Outputs:
- `runs/<experiment>/checkpoints/{best.pt,last.pt}`
- `runs/<experiment>/train_log.jsonl`
- `runs/<experiment>/config.json` and `runs/<experiment>/env.json`
- `artifacts/results.json`
- `artifacts/report.md`

### Full LibriSpeech (Download + Train + Eval)

1) Setup environment (creates `.venv` and installs `requirements.txt`):

```bash
make setup
```

2) Download + verify + extract + build manifests:

```bash
make data CONFIG=configs/librispeech_baseline.yaml
```

3) Train:

```bash
make train CONFIG=configs/librispeech_baseline.yaml
make train CONFIG=configs/librispeech_ucc_n4_sil200_ramp.yaml
make train CONFIG=configs/librispeech_ablation_no_silence.yaml
```

4) Evaluate and aggregate results across experiments, then render the markdown report:

```bash
make eval CONFIG=configs/librispeech_eval.yaml
make report
```

Direct entrypoints (what the Make targets call):
- `.venv/bin/python -m ucc_asr.pipeline --config <yaml> --stage data|train|eval`
- `.venv/bin/python -m ucc_asr.report --results artifacts/results.json --out artifacts/report.md`

## Limitations (Current State)

- The committed numbers are from the synthetic smoke dataset; they should not be compared to LibriSpeech benchmarks.
- Long-form evaluation here is synthetic concatenation of utterances, not true long recordings with real acoustic continuity.
- Decoding is greedy argmax; no LM, no beam search, no word-level timestamps/segmentation.
- UCC sampling is uniform random over the manifest; it does not control for speaker/chapter or duration distributions.
- Training is single-process (no DDP/FSDP), step-based, and uses a simple LR warmup (no cosine/plateau schedulers).

## Next Research Steps (Concrete)

1) Run LibriSpeech end-to-end and report `dev-clean/dev-other/test-clean/test-other` with the same baseline/UCC/ablation trio (`configs/librispeech_*.yaml`), plus sweep `target_n` and `rampup_fraction`.
2) Add duration-aware batching (reduce padding) and/or cap concatenated duration to stabilize CTC training at larger `N`.
3) Replace long-form synthetic eval with a real long-form benchmark (or generate session-like audio by concatenating utterances from the same speaker/chapter with realistic silences).
4) Add decoder upgrades: beam search, optional LM rescoring, and seam-aware metrics beyond a fixed `seam_window_words`.

## License

Project code: MIT (see `LICENSE`).  
LibriSpeech data: CC BY 4.0 (per OpenSLR).
