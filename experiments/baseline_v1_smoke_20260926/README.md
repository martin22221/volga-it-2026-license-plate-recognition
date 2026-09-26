# Baseline V1 — smoke runs, 2026-09-26 (NOT a baseline)

Plumbing checks on a CPU-only machine (AMD Ryzen 7 7735HS, no CUDA GPU). **These
are not Baseline V1 results, and no metric in them says anything about model
quality.** They prove that each stage runs end to end, that the torch and ONNX
graphs agree, and what the deployed pipeline costs per stage on this CPU.

| | Detector | Recogniser |
| --- | --- | --- |
| Data | 64 `synthetic:train` images, 32 `synthetic:val` | 1,024 `synthetic:train` plates, 256 `synthetic:val` |
| Epochs | 2 | 2 |
| Result | losses falling (8.16 → 6.47); val recall 0.28 → 0.53 at mAP50 0.02 | losses falling (5.96 → 3.44); OCR exact 0.00 |

No real image was read by any of these runs.

Files: `*/run.json` (full run identity, versions, hardware, config),
`*/metrics.jsonl` (per epoch), `export/parity.json` (torch vs ONNX),
`export/*.json` sidecars (preprocessing, output format, thresholds, sizes),
`export/bench_cpu_*.json` (per-stage latency, see below). The weights
(`best.pt`, `last.pt`, `*.onnx`) are not committed.

## Export parity (synthetic:val, fixed strided sample)

* Detector: 16 images, max |Δbox| 0.005 px, max |Δscore| 1.6e-5, identical
  detection count after NMS on 16/16, worst matched IoU 0.99999.
* Recogniser: 64 plates, identical raw string, type and two-line decision on
  64/64; max |Δ| ≤ 4.3e-6 at batch sizes 1, 3 and 64.

## CPU latency (ONNX Runtime 1.30, CPUExecutionProvider, this laptop — not the target)

200 `synthetic:val` images (1280×720 / 1024×768), 10 untimed warm-up images,
one timed pass each, `--max-plates 1` (every synthetic image holds exactly one
plate; the untrained detector fires on 100 boxes per image, which would
otherwise time the recogniser 100 times):

| Stage | median ms | mean ms | p95 ms |
| --- | --- | --- | --- |
| decode | 4.8 | 4.8 | 7.6 |
| detector preprocess | 18.6 | 18.0 | 22.4 |
| detector (ONNX) | 25.2 | 27.6 | 41.4 |
| detector postprocess | 9.6 | 10.0 | 12.5 |
| recogniser, both passes | 12.6 | 13.5 | 20.8 |
| validate + confidence | 0.02 | 0.02 | 0.02 |
| **pipeline total** | **72.6** | **74.9** | **102.7** |

As deployed with the smoke detector (100 plates read per image): 679 ms mean.
With a trained detector on real scenes, expect one to a few plates per image.

**This does not show that the ≤ 100 ms budget is met.** The target is an
i5-7600 with a GTX 1050 Ti. This is a different CPU and uses no GPU. Postprocessing
time also depends on how many candidates a trained detector leaves above threshold.
