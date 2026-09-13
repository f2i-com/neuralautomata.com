# Recorded experimental results

Independent CPU reference experiments. These are not measurements of `izuc/nca`, a frontier language model, or a natural-language knowledge learner. All memory tests use frozen shared weights during evaluation.

## Remote associative memory

Each main arm: 600 outer training updates, batch 64, eight random key/value records on an eight-cell periodic ring. Every third training batch appends a replacement value for the queried key at its original location. Eight-bit values and their assignments change by episode. Known key vocabulary: 16 symbols. Evaluation: 512 fixed new episodes per condition, shared across three initialization seeds. No training/validation tuning split was used for hyperparameter search; this is exploratory.

| Model | Eight records | Overwrite queried record | Sixteen records | Sixteen records in one cell | Sixteen-cell ring |
|---|---:|---:|---:|---:|---:|
| Learned 16-dimensional keys | 100.00% | 100.00% | 100.00% | 99.80% | 100.00% |
| Frozen random 16-dimensional keys | 65.82% | 69.92% | 30.53% | 40.30% | 65.76% |
| 128-state GRU control | 4.69% | 29.17% | 2.02% | 4.62% | 3.12% |
| Learned 8-dimensional keys | 55.79% | 62.89% | 20.12% | 24.41% | 58.66% |
| Exact symbolic dictionary, not trained | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

All table scores are **whole eight-bit exact recall**, not per-bit accuracy. The dictionary uses the same local relay and 128 numeric payload values per cell in its 16-symbol configuration. It receives exact one-hot addressing. It is a positive algorithmic control, not a neural training result.

The learned 16-key fast model has 385 slow parameters and 128 adaptive floats per cell. The frozen-key arm stores the same 385 coefficients but trains only 129. The GRU control has the same 128 private floats per cell but **69,313** slow parameters and a different, more expensive read/write function. The key-width-eight arm has 257 slow parameters and 64 adaptive floats per cell. Consequently this is not a parameter-, FLOP-, latency-, or tuning-matched architecture contest.

## Mechanism interventions

| Intervention, learned 16-dimensional keys | Per-bit accuracy | Exact recall |
|---|---:|---:|
| Intact memory | 100.00% | 100.00% |
| Erase all private memory | 49.76% | 0.20% |
| Disable writes | 49.76% | 0.20% |
| Erase writer cell | 50.94% | 0.39% |
| Disable neighbor relay | 49.74% | 0.39% |
| Complementary donor memory, scored against donor | 100.00% | 100.00% |
| Complementary donor memory, scored against original | 0.00% | 0.00% |

Random-value chance is 50% per bit and 1/256 (0.390625%) for all eight bits. Finite sampled controls fluctuate around these values. The reset output is deterministically all-zero; it matches one of the 512 sampled targets. The donor uses the same keys and sites with all values complemented. Because this prototype is linear in values, complementary donor predictions are expected to negate original logits. This is a provenance intervention, not independent evidence of a complex transplant capability.

Query packets are freshly initialized. Private memory is not changed by queries. Shared-parameter hashes before and after evaluation agree; shared-weight checkpoint reload reproduces evaluated scores after re-ingesting the same episode. A separately saved acquired-memory snapshot also reproduces its answer.

## Literal examples

These are the first four predetermined held-out examples for seed 0, not selected best cases.

| Key ID | Writer cell | Query cell | Written | Answer | Complementary donor | Donor answer |
|---:|---:|---:|---|---|---|---|
| 5 | 2 | 6 | `01100110` | `01100110` | `10011001` | `10011001` |
| 7 | 6 | 4 | `10110100` | `10110100` | `01001011` | `01001011` |
| 13 | 3 | 2 | `01011110` | `01011110` | `10100001` | `10100001` |
| 11 | 3 | 7 | `01110110` | `01110110` | `10001001` | `10001001` |

## Key geometry

Mean absolute off-diagonal key cosine: 0.0418 with learned keys versus 0.1999 with frozen random keys. This supports reduced representational interference, not emergence of natural-language semantics.

## Repeated overwrites

Initialize all sixteen records, then issue 256 newly observed replacement values. Each symbol keeps a fixed writer location within a stream. Probe all sixteen current values in 128 episodes at each checkpoint. There is no unseen-future target in a query and no slow-parameter learning during this stress test.

| Completed replacements | Learned 16-key exact recall | Learned 8-key exact recall |
|---:|---:|---:|
| 0 | 100.00% | 22.49% |
| 16 | 99.95% | 20.20% |
| 64 | 99.97% | 19.91% |
| 256 | 99.95% | 19.06% |

## Longer recurrent control

An additional seed-0 GRU run received 2,400 outer updates, four times the main budget. It reached 87.89% on the latest rewritten query, but only 3.32% on an arbitrary earlier association. This suggests a recency shortcut in this control under this recipe. It is not a claim that recurrent neural networks cannot implement associative memory. No broad hyperparameter search was run.

## Attention-free language example

A **separate** causal NCA byte model was trained on original template-generated English. There is no fast-memory branch in this model. It has 29,056 parameters, 32 channels per byte, radius two and eight shared updates. Its maximum dependency window is 17 bytes.

The corpus contains 4,096 combinations of eight colors, actors, verbs and objects. Training uses 3,584 distinct sentences, validation 512 held-out combinations; vocabulary and short substrings are shared. This is not a Wikipedia, TinyStories, instruction-following, or general-language benchmark.

| Training updates | Sampled validation bits per byte | Processed byte targets |
|---:|---:|---:|
| 400 | 1.7506 | 614,400 |
| 2000 | 0.4644 | 3,072,000 |

Initial untrained loss was 8.0046 bits per byte. Validation uses sixteen fixed sampled batches, including left-edge warmup positions. No sentence was selected by loss or fluency for generation. A single initialization seed was run; the longer run restarted with the same seed and data schedule rather than resuming optimizer state.

Literal output, prompt `The `, seed 0, temperature 0.3:

```text
The silver robot carries the book.
The red robot finds the box.
The green sailor carries the book.
The black sailor carries the book.
The blue robot carries the box
```

At temperature 1.0, the same trained model produces malformed words and incomplete syntax:

```text
The silves the stone.
The gold raints the stone.
The rel stores the box.
Thenple map.
The red red ines the lamp.
The purple 
```

Low-temperature template completion does not establish language understanding or a capable assistant. The supplied samples at temperatures 0.3, 0.7 and 1.0 are all retained. There is no tuned language-model baseline in this package.

## Correctness checks

......................                                                   [100%]
22 passed in 1.12s

Tests cover local nonmutating writes, write masks, outer gradients, frozen parameters, packet speed, read-only queries, per-batch isolation, memory checkpoint replay, exact dictionary overwrite, delta-gradient equivalence, prefix invariance, streaming/full-forward parity, finite language context, and language gradients. On the trained language checkpoint, maximum full-versus-streaming logit difference was 5.0067901611328125e-06, with identical argmax outputs on the checked validation prefix.

The first combined memory command was interrupted by a command timeout after completed fast runs and part of a frozen-key run. Frozen-key and GRU arms were rerun from scratch to completion. The 200-update pilot is retained. CPU wall times overlapped other runs and are **not comparative performance benchmarks**. No original-repository code, GPU run, browser integration or production service was involved.

## Repeating the trained language rule more times

Same 2,000-update checkpoint, no retraining. The training horizon was eight cellular iterations.

| Inference iterations | Maximum context, bytes | Sampled validation bits per byte |
|---:|---:|---:|
| 1 | 3 | 3.6238 |
| 2 | 5 | 2.1891 |
| 4 | 9 | 0.9012 |
| 8 | 17 | 0.4644 |
| 16 | 33 | 0.6635 |
| 32 | 65 | 1.6321 |

Additional iterations beyond the trained horizon increased loss in this example, despite a larger theoretical receptive field. This is an inference-depth ablation, not a comparison of separately trained models.
