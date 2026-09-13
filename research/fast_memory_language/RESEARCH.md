# Cellular Fast Memory and Non-Transformer Language Modeling

## 1. Conclusion and scope

A language model does not have to be a transformer. A shared neural cellular rule can also be trained to predict language. The substantive questions are whether it learns useful dependencies, whether training and generation are efficient, and whether adaptation preserves rather than corrupts earlier information. Non-transformer feasibility is established by other architecture families; it does not establish that a particular NCA design is competitive.

This package provides two separate pieces of executed evidence. The first integrates learned representations with local adaptive matrices and a cellular query protocol. The second implements and trains a strictly causal, attention-free byte model. They are not yet one model, and their combination has not been evaluated. The memory task uses symbolic records; the language task uses a small generated grammar rather than a natural corpus.

The implementation is independent of `izuc/nca`. A public request to that repository returned 404 during this continuation; no original source, current commit, browser implementation or original checkpoint was obtained. The normalized delta primitive from the previous independent lab is reused unchanged. Neither old H-NCAM reports nor the previous lab's model weights are treated as evidence about these new models.

New numerical claims are supported by `RESULTS.md`, `SUMMARY.json`, per-run metrics and logs, and retained checkpoints. Literature observations are cited below. All runs were CPU FP32. No GPU speed, frontier-scale capability or production readiness is claimed.

## 2. What the new memory experiment asks

The previous lab checked a standalone delta-writing operation and separately trained shared NCA parameters online. The next missing mechanism was a model that learns useful representations for fast writes, then acquires new episode information while its shared parameters are frozen.

The new question is: can a value be written into a cell's adaptive memory, queried from a different cell after the original input is unavailable, and replaced without destroying unrelated associations?

A fixed one-directional ring is used to isolate this question. Transport is deliberately supplied, not learned. Every cell uses the same function and has its own private adaptive matrix. A public query packet moves to one neighboring cell per synchronous update. It collects local reads and returns to the query origin after one complete lap.

The current system is therefore best described as a **structured cellular fast-memory prototype**, not a demonstration of unconstrained self-organization. Routing, write timing, memory layout and the analytic learning law are supplied. Training learns key embeddings, value encoding, output decoding and a logit scale. Distinguishing those contributions prevents a useful control experiment from being misrepresented as autonomous architectural invention.

### Episode protocol

Training uses eight cells, sixteen available key symbols and eight-bit values. An episode assigns eight distinct symbols to randomly selected writer cells and random values. Multiple symbols may reside in the same cell. The query location is explicitly different from the queried symbol's writer. Each third training batch appends an overwrite of the queried symbol at the same writer, using its complemented value.

The observation interface provides a value only during a write event. Later queries receive a key and query location, never the target value. Fast memory is empty at the start of each independent training episode. Query packets are freshly reset for each question, while previously acquired private matrices are preserved.

Evaluation uses new randomized assignments, layouts and combinations. Individual symbols and eight-bit values may have appeared in training. This is novel episode binding, not a test of unseen symbol identities, arbitrary natural-language keys or unbounded vocabulary.

## 3. Exact cellular memory architecture

Let cell i contain a private matrix A_i of shape V by K. In the main model V=8 and K=16. Eight cells therefore contain 1,024 adaptive FP32 values, or 4 KiB, excluding packets, indices, weights and temporary execution buffers.

For a known key symbol c, the shared encoder produces a normalized vector:

```text
k = normalize(Embedding(c))
v = W_value * (2 * observed_bits - 1)
```

An explicit local observation mask identifies the writer. Only that cell changes its matrix:

```text
error = v - A_i @ k
A_i_next = A_i + outer(error, k) / (epsilon + dot(k, k))
```

This is a normalized gradient step on a local squared read error. It is implemented by differentiable tensor arithmetic. Outer training can therefore differentiate through the write to learn its input representations. The rank-one update and its rate of one are supplied, not discovered or learned by the model.

The reuse of an adaptive model as recurrent memory connects to TTT layers, which frame hidden-state updates as learning steps. Differentiable plasticity separately establishes that initial training can optimize mechanisms that later acquire new associations. Neither paper is evidence that this specific cellular protocol is novel or broadly successful.[^1][^2]

### Local query communication

A packet contains the encoded key q, an accumulated value a and a validity flag. For one synchronous update:

```text
incoming_i = previous_packet_at_left_neighbor(i)
a_i_next = incoming_i.accumulator + A_i @ incoming_i.key
packet_i_next = [incoming_i.key, a_i_next, incoming_i.valid]
```

The validity flag masks absent packets. After N updates on an N-cell ring, the packet returns to its origin. The answer is the shared linear decoder of its accumulated value, multiplied by a learned scale. The implementation uses local shifts and reads; there is no global lookup of the writer's memory at the query call.

The main model has 385 shared slow coefficients: 256 key-embedding coefficients, 64 value-encoder coefficients, 64 decoder coefficients and one scale. The private adaptive matrices are runtime memory, not counted as slow parameters. Both quantities must be reported.

### An important algebraic limitation

After one complete lap the accumulated read equals:

```text
sum_i(A_i @ q) = (sum_i A_i) @ q
```

Consequently the read stage alone does not demonstrate a capability requiring a cellular layout. It could be evaluated from a summed matrix. Local writes at different cells do have separate update states, but there is no learned, selective routing in this model. An exact dictionary also solves the task.

This equivalence is useful rather than embarrassing: it identifies exactly which claims the current experiment cannot support. A later model needs a reason for spatial allocation, nonlinear communication or selective querying before cellular organization becomes an efficiency or capability argument.

### Computation is not automatically sparse

The dense reference evaluates all cells each round, even though one packet is active. Work per round is linear in cell count at fixed widths. A complete lap needs N rounds, so the current dense implementation has quadratic work per completed ring query, with linear communication latency. Multiplying inactive computations by zero does not save their arithmetic.

An event-sparse implementation could avoid some inactive work, but it has not been implemented or profiled. The current ring is a semantic test harness, not a proposed high-throughput long-context language engine.

## 4. How the model learns to use fast writes

The outer loop samples fresh episodes, ingests observed records through differentiable local writes, runs a query, and minimizes binary cross-entropy against the episode's correct value. It uses AdamW, learning rate 0.003, zero weight decay, gradient clipping at one, 600 updates and batch 64. The main comparison uses three initialization seeds and fixed separate evaluation streams.

```python
optimizer.zero_grad(set_to_none=True)
private_memory = ingest(model, observed_episode)
logits = model.query(private_memory, query_key, query_cell)
loss = binary_cross_entropy_with_logits(logits, evaluator_target)
loss.backward()
optimizer.step()
```

Only this outer training phase changes shared weights. During evaluation, weights remain frozen and fingerprints are checked before and after. New observed records still update the adaptive matrices. This is a limited learning-to-learn construction: the representation is optimized to make a specified learning law useful. It is not learned objective discovery, learned source verification or a learned optimizer policy.

The write contract is event-based. Extra query or computation rounds do not write the same observation repeatedly. Separating observation events from internal iterations prevents renderer frame rate or reasoning depth from accidentally changing memory strength.

## 5. What the experiment found

The complete numerical tables are generated from raw records in `RESULTS.md`; the most important findings are summarized here.

The sixteen-dimensional learned-key model achieved 100% whole-eight-bit recall on the sampled ordinary, overwrite and sixteen-record tests across the three seeds. With all sixteen records concentrated at one writer, the mean was 99.80%. The same weights also answered queries on a sixteen-cell ring. That size result is supported by the supplied relay and position-shared encoders, not evidence of emergent size-general planning.

Resetting all adaptive matrices, disabling writes or erasing the relevant writer removes the relevant episode information and reduces performance to chance-level scores. Query-only execution does not change private memory or shared weights. Saving acquired matrices together with the shared checkpoint reproduces their answers after reload; saving shared weights alone would not preserve those newly acquired records.

The learned representations reduce interference. Mean absolute cosine between different keys fell to approximately 0.042; frozen random keys remained near 0.200. The frozen-key model reached 65.82% exact recall on eight records and 30.53% on sixteen. It uses the same adaptive shape and supplied write law, but its value encoder and decoder cannot compensate fully for arbitrary key cross-talk.

Halving key width to eight reduces both adaptive capacity and encoder parameter count. The resulting model reached 55.79% exact recall on eight records and 20.12% on sixteen. This is an important limitation, not proof that capacity is the only factor: representation size, key geometry and trainable count all change together.

After initializing sixteen records and observing 256 subsequent replacements, the main learned-key arm retained approximately 99.95% exact recall of the current values, averaged across seeds. Its writers remain fixed by key during the stream. Cross-site contradictory versions, eviction, unknown keys and user/source boundaries were not tested.

### Controls and interpretive limits

The exact dictionary achieved perfect results on the recorded conditions. Its keys are exact symbolic addresses, supplied rather than learned. It is a positive algorithmic baseline with the same numeric payload width in the sixteen-key comparison. Perfect neural recall does not establish an advantage over it.

A per-cell GRU control has the same 128 private floats and public relay, but 69,313 shared parameters and a different read/write function. Under 600 updates, it performed poorly on arbitrary earlier associations. A further seed-0 run with 2,400 updates reached 87.89% on the latest overwritten value while remaining at 3.32% on an arbitrary earlier record. This suggests a recency shortcut under this training recipe, not an inability of recurrent networks to implement memory.

The comparison is not parameter-, FLOP-, latency- or tuning-matched. It is evidence that the chosen delta-memory structure fits this task particularly directly. A carefully tuned recurrent baseline or a different state representation may behave differently. The supplied protocol also resembles a dictionary much more closely than open-ended language comprehension.

Donor-memory tests use complementary values with identical key/site layout. This model is linear in the written values, so complementary memory predictably negates its logits. The intervention verifies the path carrying the information, but is not a distinct sophisticated transplant capability.

## 6. Can an NCA be a language model without being a transformer?

Yes. A language model is a model of a language sequence distribution. A transformer is one way to parameterize it. Mamba provides an attention-free selective state-space language-model architecture. RWKV-7 provides a recurrent alternative using dynamic matrix-state updates. These are non-NCA precedents showing that transformer blocks are not required.[^3][^4]

There is also older direct precedent for attention-free convolutional language models: gated convolutional networks were evaluated on large language datasets without a transformer architecture. Reusing a convolutional rule iteratively adds a cellular/recurrent structural choice; it does not create language modeling from a wholly unrelated objective.[^5]

An August 2026 preprint, TextNCA, directly investigates cellular-style language modeling. Its main architecture uses local attention, so it is not an attention-free precedent. It reports worse WikiText-103 perplexity than its transformer controls and a bounded benefit from iteration. Its role here is a relevant caution about locality and weight sharing, not proof that our approach will compete.[^6]

Use operational descriptions rather than architectural branding. Token embeddings, gates, residual updates and vocabulary softmax are not exclusive to transformers. Matrix-memory formulations can have connections to linear attention without being conventional dense self-attention blocks. The proposed goal is a clear computation graph with measured properties, not winning an argument about what to call it.[^1][^7]

## 7. Executed attention-free byte language model

A separate implementation assigns one cell to each byte position. It contains a byte embedding, a causal radius-two convolution, a local input anchor, a gated state update and an output distribution over 256 bytes. The same rule is reused eight times. No attention, future-facing convolution, spatial normalization, pooling or pretrained language encoder is present.

For position i and cellular iteration k:

```text
anchor_i = projection(tanh(byte_embedding(x_i)))
h_i = SiLU(local_convolution(s[i-2], s[i-1], s[i]) + anchor_i)
g_i = sigmoid(gate_projection(h_i))
s_i_next = (1-g_i)*s_i + g_i*tanh(candidate_projection(h_i))
next_byte_logits_i = output_projection(s_i_final)
```

Initial states are bounded embeddings. The sigmoid mixture keeps each coordinate in a bounded interval under the stated starting conditions. It does not ensure correct language or convergence. There are 32 channels per cell, a 64-unit shared hidden projection and 29,056 total parameters.

Training minimizes next-byte cross-entropy:

```text
input:   x_0, x_1, ..., x_(T-1)
target:  x_1, x_2, ..., x_T
```

Every output may use the current byte and its past, but never the future target byte. Shared-rule gradients accumulate through all eight cellular iterations. Unlike the memory experiment, this language example does not change weights or fast memory during inference.

### Actual text experiment

The corpus consists of 4,096 original template combinations of colors, actors, verbs and objects. There are 3,584 training sentences and 512 disjoint validation combinations. Vocabulary and short substrings are shared; this is not a natural-language generalization benchmark.

With 2,000 updates at batch sixteen and length ninety-six, the model processes 3,072,000 next-byte targets. Sampled validation loss falls from 8.0046 to 0.4644 bits per byte. The shorter 400-update run and its malformed output are retained. Both runs use the same seed and schedule; the longer run restarts rather than resuming optimizer state.

At temperature 0.3, an actual sample begins:

```text
The silver robot carries the book.
The red robot finds the box.
The green sailor carries the book.
```

At temperature 1.0, the same model produces malformed words and incomplete syntax. Both outputs are in the package. The valid low-temperature sentences demonstrate local template learning, not semantic understanding, document acquisition or an assistant. A single seed and no tuned LM baseline are insufficient for broad performance claims.

### The context bottleneck

For a causal radius r repeated K times, the maximum receptive field is `1+r*K` input positions. The default model therefore sees at most **17 bytes**, even when given a much longer tensor. More input cells do not increase that bound at fixed depth. Many words, dependencies and conversational facts lie outside it.

A trained-model sweep illustrates another limitation. Its validation loss is 0.4644 bits per byte at the trained eight iterations, 0.6635 at sixteen, and 1.6321 at thirty-two. More computation exposes a larger theoretical neighborhood but makes this checkpoint's predictions worse. Variable inference depth must be trained and evaluated; it cannot be assumed to provide extra reasoning.

### Correct incremental decoding

The incremental implementation caches the last r predecessor states **at each iteration stage**, not only the final state. That is the state needed to compute a newly appended byte exactly as full causal evaluation would. Reusing final states for every internal iteration would implement a different recurrence.

Tests check prefix invariance, maximum dependency range and full-forward/streaming agreement. On a checked validation prefix of the trained checkpoint, the largest absolute logit discrepancy was about 5e-6, with identical argmax outputs. This is a numerical parity result in the recorded CPU environment, not universal bitwise equality.

The cache is constant-size for fixed r, K and state width. That does not mean unlimited retained context: it is an efficient implementation of the same finite-context computation. It must reset at independent stream boundaries.

## 8. A plausible combined architecture

Combining the two demonstrations requires additional design and testing. The current memory relay must not simply be placed on a teacher-forced token ring: wrapping messages around positions would allow a prediction to read future tokens.

Two distinct designs are plausible.

### Token-lattice design

Keep one cell per token with strictly causal communication. Add fast-memory operations that only use already available prefix information. A hierarchy may summarize completed earlier chunks and feed those summaries to later positions, but incomplete chunks must not leak their future tokens into earlier predictions.

Dilation can expand the graph-distance receptive field without increasing degree at every update. It changes the locality notion and may skip useful intermediate structure. Hierarchy, dilation and memory should be tested separately rather than added together. A small GRU and an ordinary causal convolutional model are necessary matched-data controls before scaling.

### Persistent-substrate design

Keep a fixed lattice of computational cells while tokens arrive over environment time. Inject one observed token through a documented input port, perform local updates and read the next-token prediction from a documented output port. Cells may communicate in both spatial directions because no future token has arrived yet. Private matrices retain selected observations across token events.

This is a nonlinear recurrent system whose hidden state is spatially structured. It offers a cleaner route to persistent learning, but creates a sequential training path, communication delays and finite-state capacity. A fixed ring relay at every token would be a particularly costly default. The read/write policy, port allocation and communication budget must be learned or justified by a task.

Neither design has yet been trained as a combined cellular fast-memory language model. They are concrete hypotheses for the next stage, not descriptions of delivered functionality.

## 9. Causal adaptation while reading language

A self-supervised learning signal can come from subsequently observed text. It must be associated with the prediction that preceded it. A safe experimental sequence is:

```text
Read observed prefix through x_t.
Predict x_(t+1) and record the prediction.
Receive x_(t+1) from the allowed data stream.
Score the recorded prediction.
Perform any permitted fast update using now-available evidence.
Predict later tokens.
```

Another legitimate objective can write a representation of already observed x_t before predicting x_(t+1). What is forbidden by the causal modeling contract is allowing x_(t+1) or later content into the state used to score the earlier prediction. Training on a whole chunk and then scoring that same chunk with its final adapted memory is a common way to violate this contract.

TTT and Gated DeltaNet supply relevant mechanisms for learning or modifying matrix state during sequence processing, including approaches intended to reduce sequential training overhead. Their published mechanisms are not automatic plug-ins that eliminate the causal and performance constraints of an arbitrary nonlinear cellular loop.[^1][^7]

Slow parameter learning should remain distinct from fast adaptation. A supplied document can provide temporary evidence for questions about it; that does not establish that its claims are correct, that they have been consolidated into weights, or that they should affect unrelated users. Preserve source/version information in a separate evidence store and evaluate each persistence level independently.

## 10. Next development sequence

First replace the finite learned symbol table with a compositional key encoder. Train on varied structured names and property sequences, then test combinations and encodings not seen in outer training. Preserve exact symbolic addressing and capacity-matched state baselines. This tests whether the learned representation can generalize rather than merely separate sixteen known symbols.

Second remove the full-lap sum bottleneck only if a benchmark needs something more selective. Compare learned destination selection, nonlinear local read/response rules and one fixed hierarchical relay. Count all selection and communication work. Require a gain over a global fast matrix or exact dictionary under an explicit budget before treating the spatial substrate as useful.

Third build a stronger language baseline before adding live adaptation. A proposed first target is a small byte model trained on licensed, deduplicated natural text, compared with a GRU and causal convolutional control under common data and tuning budgets. Report bits per byte, generation latency, memory, long-distance retrieval and exact input provenance. A larger parameter count alone is not the next milestone.

Fourth integrate causal fast memory into the best-understood sequence design. Use synthetic project records whose facts are randomized per episode, then ask paraphrased and compositional questions after removing the original text. Distinguish absent facts, updated facts and conflicting sources. Reset working state, adaptive matrices and shared weights independently to locate what retains the information.

Finally examine durable consolidation and chat behavior. General text prediction, following instructions, reliable factual responses and continuous retention are separate objectives. They should not be inferred from a few grammatical continuations. A billion-parameter run would be premature until a smaller model shows a useful quality-versus-compute result or a novel adaptation property.

The current experiments make the proposal more concrete: local adaptive memory can acquire new associations, and an attention-free cellular rule can learn a text distribution. They also expose the next obstacles: supplied routing, finite addressing capacity, nonlinear retrieval, causal long-context integration and stable computation beyond the training horizon.

## Sources

Literature checked on 13 September 2026. New experimental results are supported by this package's retained raw files, not by the literature.

[^1]: Sun et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*. arXiv:2407.04620v4, revised 31 August 2025; original submission July 2024. `https://arxiv.org/html/2407.04620v4`
[^2]: Miconi, Stanley and Clune. *Differentiable plasticity: training plastic neural networks with backpropagation*. ICML / PMLR 80, 2018. `https://proceedings.mlr.press/v80/miconi18a.html`
[^3]: Gu and Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752v2, revised May 2024. `https://arxiv.org/abs/2312.00752`
[^4]: Peng et al. *RWKV-7 “Goose” with Expressive Dynamic State Evolution*. arXiv:2503.14456, March 2025. `https://arxiv.org/abs/2503.14456`
[^5]: Dauphin et al. *Language Modeling with Gated Convolutional Networks*. ICML / PMLR 70, 2017. `https://proceedings.mlr.press/v70/dauphin17a.html`
[^6]: Mittal et al. *TextNCA: Neural Cellular Automata for Language Modeling via Hierarchical Local Attention*. arXiv:2608.02050v1, 3 August 2026, preprint. `https://arxiv.org/html/2608.02050v1`
[^7]: Yang, Kautz and Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. arXiv:2412.06464; version 3 inspected. `https://arxiv.org/html/2412.06464v3`
