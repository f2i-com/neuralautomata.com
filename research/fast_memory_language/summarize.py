"""Build recorded-results tables from executed runs, not manually typed scores."""
import json,statistics
from pathlib import Path

def groups():
    out={}
    for root,v in [('results','fast'),('results','frozen_keys'),('results','gru'),('capacity_results','fast_k8')]:
        out[v]=[json.loads(p.read_text()) for p in sorted(Path(root).glob(f'{v}_seed*/metrics.json'))]
        if len(out[v])!=3:raise RuntimeError(f'Expected three completed {v} runs')
    return out

def main():
    g=groups();summary={}
    for v,runs in g.items():
        summary[v]={'seeds':[r['seed'] for r in runs],'counts':runs[0]['counts'],'metrics':{}}
        for metric in runs[0]['metrics']:
            summary[v]['metrics'][metric]={}
            for field in runs[0]['metrics'][metric]:
                vals=[r['metrics'][metric][field] for r in runs]
                summary[v]['metrics'][metric][field]={'mean':statistics.mean(vals),'min':min(vals),'max':max(vals)}
    Path('SUMMARY.json').write_text(json.dumps(summary,indent=2))
    def val(v,k,f='exact_accuracy'):return summary[v]['metrics'][k][f]['mean']
    def pct(x):return f'{100*x:.2f}%'
    lines=['# Recorded experimental results','',
      'Independent CPU reference experiments. These are not measurements of `izuc/nca`, a frontier language model, or a natural-language knowledge learner. All memory tests use frozen shared weights during evaluation.','',
      '## Remote associative memory','',
      'Each main arm: 600 outer training updates, batch 64, eight random key/value records on an eight-cell periodic ring. Every third training batch appends a replacement value for the queried key at its original location. Eight-bit values and their assignments change by episode. Known key vocabulary: 16 symbols. Evaluation: 512 fixed new episodes per condition, shared across three initialization seeds. No training/validation tuning split was used for hyperparameter search; this is exploratory.','',
      '| Model | Eight records | Overwrite queried record | Sixteen records | Sixteen records in one cell | Sixteen-cell ring |',
      '|---|---:|---:|---:|---:|---:|']
    names={'fast':'Learned 16-dimensional keys','frozen_keys':'Frozen random 16-dimensional keys','gru':'128-state GRU control','fast_k8':'Learned 8-dimensional keys'}
    for v in names:
        lines.append('| '+names[v]+' | '+' | '.join(pct(val(v,k)) for k in ['in_distribution','overwrite','more_records','concentrated','larger_ring'])+' |')
    lines+=['| Exact symbolic dictionary, not trained | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |','',
      'All table scores are **whole eight-bit exact recall**, not per-bit accuracy. The dictionary uses the same local relay and 128 numeric payload values per cell in its 16-symbol configuration. It receives exact one-hot addressing. It is a positive algorithmic control, not a neural training result.','',
      'The learned 16-key fast model has 385 slow parameters and 128 adaptive floats per cell. The frozen-key arm stores the same 385 coefficients but trains only 129. The GRU control has the same 128 private floats per cell but **69,313** slow parameters and a different, more expensive read/write function. The key-width-eight arm has 257 slow parameters and 64 adaptive floats per cell. Consequently this is not a parameter-, FLOP-, latency-, or tuning-matched architecture contest.','',
      '## Mechanism interventions','',
      '| Intervention, learned 16-dimensional keys | Per-bit accuracy | Exact recall |','|---|---:|---:|']
    for k,title in [('in_distribution','Intact memory'),('reset_fast_memory','Erase all private memory'),('writes_disabled','Disable writes'),('erase_source_cell','Erase writer cell'),('communication_disabled','Disable neighbor relay'),('transplant_vs_donor','Complementary donor memory, scored against donor'),('transplant_vs_original','Complementary donor memory, scored against original')]:
        lines.append(f'| {title} | {pct(val("fast",k,"bit_accuracy"))} | {pct(val("fast",k))} |')
    lines+=['',
      'Random-value chance is 50% per bit and 1/256 (0.390625%) for all eight bits. Finite sampled controls fluctuate around these values. The reset output is deterministically all-zero; it matches one of the 512 sampled targets. The donor uses the same keys and sites with all values complemented. Because this prototype is linear in values, complementary donor predictions are expected to negate original logits. This is a provenance intervention, not independent evidence of a complex transplant capability.','',
      'Query packets are freshly initialized. Private memory is not changed by queries. Shared-parameter hashes before and after evaluation agree; shared-weight checkpoint reload reproduces evaluated scores after re-ingesting the same episode. A separately saved acquired-memory snapshot also reproduces its answer.','',
      '## Literal examples','',
      'These are the first four predetermined held-out examples for seed 0, not selected best cases.','',
      '| Key ID | Writer cell | Query cell | Written | Answer | Complementary donor | Donor answer |','|---:|---:|---:|---|---|---|---|']
    for x in g['fast'][0]['examples']:
        lines.append(f"| {x['symbol']} | {x['writer']} | {x['reader']} | `{x['written']}` | `{x['answer']}` | `{x['donor']}` | `{x['transplanted_answer']}` |")
    lines+=['','## Key geometry','',
      f"Mean absolute off-diagonal key cosine: {val('fast','key_geometry','mean_abs_offdiag'):.4f} with learned keys versus {val('frozen_keys','key_geometry','mean_abs_offdiag'):.4f} with frozen random keys. This supports reduced representational interference, not emergence of natural-language semantics.",'',
      '## Repeated overwrites','',
      'Initialize all sixteen records, then issue 256 newly observed replacement values. Each symbol keeps a fixed writer location within a stream. Probe all sixteen current values in 128 episodes at each checkpoint. There is no unseen-future target in a query and no slow-parameter learning during this stress test.','',
      '| Completed replacements | Learned 16-key exact recall | Learned 8-key exact recall |','|---:|---:|---:|']
    stress=json.loads(Path('stress_results.json').read_text())
    for t in range(4):
        a=[stress[f'results/fast_seed{s}']['curves'][t]['exact_accuracy'] for s in range(3)]
        b=[stress[f'capacity_results/fast_k8_seed{s}']['curves'][t]['exact_accuracy'] for s in range(3)]
        lines.append(f"| {stress['results/fast_seed0']['curves'][t]['overwrites']} | {pct(statistics.mean(a))} | {pct(statistics.mean(b))} |")
    extended=json.loads(Path('extended_control/gru_seed0/metrics.json').read_text())
    lines+=['','## Longer recurrent control','',
       f"An additional seed-0 GRU run received 2,400 outer updates, four times the main budget. It reached {pct(extended['metrics']['overwrite']['exact_accuracy'])} on the latest rewritten query, but only {pct(extended['metrics']['in_distribution']['exact_accuracy'])} on an arbitrary earlier association. This suggests a recency shortcut in this control under this recipe. It is not a claim that recurrent neural networks cannot implement associative memory. No broad hyperparameter search was run.",'',
       '## Attention-free language example','',
       'A **separate** causal NCA byte model was trained on original template-generated English. There is no fast-memory branch in this model. It has 29,056 parameters, 32 channels per byte, radius two and eight shared updates. Its maximum dependency window is 17 bytes.','',
       'The corpus contains 4,096 combinations of eight colors, actors, verbs and objects. Training uses 3,584 distinct sentences, validation 512 held-out combinations; vocabulary and short substrings are shared. This is not a Wikipedia, TinyStories, instruction-following, or general-language benchmark.','',
       '| Training updates | Sampled validation bits per byte | Processed byte targets |','|---:|---:|---:|']
    for folder in ['language_toy','language_toy_longer']:
        r=json.loads(Path(folder,'metrics.json').read_text());lines.append(f"| {r['training_steps']} | {r['after']['sampled_validation_bits_per_byte']:.4f} | {r['processed_byte_targets']:,} |")
    r=json.loads(Path('language_toy_longer/metrics.json').read_text())
    lines+=['',f"Initial untrained loss was {r['before']['sampled_validation_bits_per_byte']:.4f} bits per byte. Validation uses sixteen fixed sampled batches, including left-edge warmup positions. No sentence was selected by loss or fluency for generation. A single initialization seed was run; the longer run restarted with the same seed and data schedule rather than resuming optimizer state.",'',
       'Literal output, prompt `The `, seed 0, temperature 0.3:','', '```text',json.loads(Path('language_toy_longer/temperature_samples.json').read_text())['0.3'],'```','',
       'At temperature 1.0, the same trained model produces malformed words and incomplete syntax:','', '```text',r['sample'],'```','',
       'Low-temperature template completion does not establish language understanding or a capable assistant. The supplied samples at temperatures 0.3, 0.7 and 1.0 are all retained. There is no tuned language-model baseline in this package.','',
       '## Correctness checks','',
       Path('tests_result.txt').read_text().strip(),'',
       'Tests cover local nonmutating writes, write masks, outer gradients, frozen parameters, packet speed, read-only queries, per-batch isolation, memory checkpoint replay, exact dictionary overwrite, delta-gradient equivalence, prefix invariance, streaming/full-forward parity, finite language context, and language gradients. On the trained language checkpoint, maximum full-versus-streaming logit difference was '+str(json.loads(Path('language_toy_longer/cache_parity.json').read_text())['max_logit_abs_difference'])+', with identical argmax outputs on the checked validation prefix.','',
       'The first combined memory command was interrupted by a command timeout after completed fast runs and part of a frozen-key run. Frozen-key and GRU arms were rerun from scratch to completion. The 200-update pilot is retained. CPU wall times overlapped other runs and are **not comparative performance benchmarks**. No original-repository code, GPU run, browser integration or production service was involved.']
    lines += ['', '## Repeating the trained language rule more times', '',
      'Same 2,000-update checkpoint, no retraining. The training horizon was eight cellular iterations.', '',
      '| Inference iterations | Maximum context, bytes | Sampled validation bits per byte |', '|---:|---:|---:|']
    for row in json.loads(Path('language_toy_longer/iteration_sweep.json').read_text()):
        lines.append(f"| {row['iterations']} | {row['max_context_bytes']} | {row['sampled_validation_bits_per_byte']:.4f} |")
    lines += ['', 'Additional iterations beyond the trained horizon increased loss in this example, despite a larger theoretical receptive field. This is an inference-depth ablation, not a comparison of separately trained models.']
    Path('RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
