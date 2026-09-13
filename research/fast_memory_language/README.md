# Fast-Memory Cellular and Language Lab

This package contains **two separate research models**, not a combined online-learning chatbot:

1. A structured cellular associative-memory model with learned key/value representations, local delta writes and a fixed nearest-neighbor ring relay. It acquires new associations while shared weights remain frozen.
2. A strictly causal, attention-free byte language model with a shared recurrent convolutional rule and verified incremental decoding. Its supplied weights were trained only on original template-generated text.

`RESEARCH.md` explains the architecture, evidence, limitations and route to a larger non-transformer language model. `RESULTS.md` contains recorded tables and literal outputs; `SUMMARY.json` and per-run JSON/JSONL contain the underlying data.

These are independent examples, not modifications to `https://github.com/izuc/nca`. The prior lab's checked delta-write routine is retained unchanged in `legacy/fast_memory.py`. Other source in this package is a new implementation. No previous NCA checkpoint or pretrained language-model weights were used to initialize these models.

## Environment

Executed on CPU using Python 3.13.5 and PyTorch 2.10.0+cpu, FP32 and two intra-op threads. `results/environment.json` records the exact interpreter/platform. Run times overlapped other experiments and are not comparative benchmarks. No GPU, browser, mixed-precision or production deployment testing was performed.

From a terminal in this directory:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Use a suitable PyTorch installation for your machine. The code was verified with the recorded CPU build; matching seeds does not guarantee identical arithmetic across devices or releases. A virtual environment is recommended.

## Replay actual learned behavior

```bash
python replay_memory.py results/fast_seed0/model.pt
python generate_language.py language_toy_longer/model.pt --prompt "The " --temperature 0.3 --bytes 160
```

These commands are single-line commands and work unchanged in PowerShell. The language sample is deliberately labeled as template generation, not general language understanding.

The memory replay creates fresh random assignments, freezes all shared parameters, writes the observed values at local cells and sends a query from another cell. `replay_memory.py` prints the observed value and the answer; the former is evaluator-only during the query.

## Reproduce the memory runs

```bash
python run_memory.py --variants fast frozen_keys gru --seeds 0 1 2 --steps 600 --out results
python run_memory.py --variants fast --seeds 0 1 2 --key-dim 8 --steps 600 --out capacity_results
python run_memory.py --variants gru --seeds 0 --steps 2400 --out extended_control
python stress_memory.py
```

The default run uses 16 known key symbols and eight-bit values. Key identities are not unseen at evaluation; their episode-specific assignments, sites and value combinations are newly generated. Each private fast matrix has 128 floats in the main arm. Ring size is eight in training and eight or sixteen in evaluation.

The recurrent control has matched private-state size and the same fixed relay, but has many more slow parameters and different computational cost. It is not a tuned, compute-matched architecture comparison. The exact dictionary is an algorithmic positive control, not a trained neural model.

## Reproduce the language demonstration

```bash
python make_toy_text.py
python train_language.py --train toy_text/train.txt --validation toy_text/validation.txt --steps 400 --length 96 --out language_toy
python train_language.py --train toy_text/train.txt --validation toy_text/validation.txt --steps 2000 --length 96 --out language_toy_longer
python extra_checks.py
python summarize.py
```

The longer run restarts from the same initialization and data schedule. `model.pt` files contain configuration and inference weights, not full optimizer/RNG training-resume checkpoints.

For real text, provide two explicit separate local UTF-8 files:

```bash
python train_language.py --train your_train.txt --validation your_validation.txt --steps 1000 --out real_text_run
```

Split by independent documents before concatenation, deduplicate them, and preserve dataset provenance. The script rejects identical files but does not implement a full deduplication or licensing pipeline. It uses bytes, so cross-entropy divided by log(2) is bits per byte, not bits per BPE token or Unicode character. All validation scores are fixed sampled windows, not exhaustive corpus likelihood.

The default model has radius two and eight iterations. **Its maximum context is only 17 bytes**, despite accepting longer input tensors. A larger tensor or incremental cache does not remove this limit. Its `--iterations` and `--radius` options change this tradeoff, but extra iterations on an already-trained model are not guaranteed to improve it. See the recorded depth sweep.

`train_language.py --device cuda` exists as a code path, but CUDA execution was not tested. Memory scripts currently run on CPU.

## State and checkpoint contract

Shared slow weights and acquired private matrices are different objects. Saving slow weights alone does not save newly ingested facts. `results/acquired_memory_snapshot.pt` shows an explicit snapshot of shared weights, acquired matrices and a pending query. The evaluator targets are stored separately in a JSON check, not inside that runtime snapshot.

Query packets are newly initialized for each question and never write private memory. Writes require an observed value at an explicit cell. A repeated query is not a new learning event. The demonstration has no confidence/unknown-key mechanism, source trust system or competing-version resolution across different writers.

## File map

- `cellular_memory.py`, `tasks.py`, `run_memory.py`: local associative learner, task protocol and outer training.
- `legacy/fast_memory.py`: reused checked normalized delta primitive.
- `stress_memory.py`, `replay_memory.py`: repeated-overwrite testing and fresh-episode replay.
- `causal_lm.py`, `train_language.py`, `generate_language.py`: separate causal byte model, training and generation.
- `make_toy_text.py`: original finite-grammar data generator; train/validation sentence combinations are disjoint.
- `extra_checks.py`, `summarize.py`: saved-state replay, language decoding/depth checks and table generation.
- `tests/test_lab.py`: 22 semantic/gradient/causality tests.
- `pilot`, `results`, `capacity_results`, `extended_control`, `language_toy`, `language_toy_longer`: retained raw runs.

The fixed ring relay is a supplied communication scaffold, not learned routing. Its complete query accumulates linear local reads, and can algebraically be collapsed into a read of summed matrices. Dense execution touches every cell each round, so a full ring query has quadratic work in ring size. This is a correctness/learning test, not a scalable long-context inference engine.
