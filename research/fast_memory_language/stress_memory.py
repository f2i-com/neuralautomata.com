"""Frozen-parameter repeated updates to known symbolic addresses.

Writers for each key remain fixed within a stream. No cross-site version
resolution is claimed. Query values are observed before writes; later queries
contain only a key. Accuracy is measured after adaptation, not before seeing a
new independent random value.
"""
from pathlib import Path
import json
import torch
from cellular_memory import CellularMemory,MemoryConfig
from run_memory import score,fingerprint

@torch.no_grad()
def evaluate(path,batch=128):
    saved=torch.load(path,weights_only=True);m=CellularMemory(MemoryConfig(**saved['config']));m.load_state_dict(saved['model']);m.eval()
    before=fingerprint(m);g=torch.Generator().manual_seed(810021);b=batch;n=8;s=16
    places=torch.randint(n,(b,s),generator=g);table=torch.randint(0,2,(b,s,8),generator=g).float()
    A=m.initial_memory(b,n);idx=torch.arange(b)
    for key in range(s):
        ids=torch.full((b,),key,dtype=torch.long);A=m.write_event(A,ids,table[:,key],places[:,key])
    results=[]
    for event in range(257):
        if event in (0,16,64,256):
            # Probe all 16 current records with frozen memory; separate readouts.
            all_logits=[];all_targets=[]
            for key in range(s):
                ids=torch.full((b,),key,dtype=torch.long)
                origin=(places[:,key]+torch.randint(1,n,(b,),generator=g))%n
                all_logits.append(m.query(A,ids,origin));all_targets.append(table[:,key])
            z=torch.cat(all_logits);y=torch.cat(all_targets)
            results.append({'overwrites':event,**score(z,y),'memory_rms':A.double().square().mean().sqrt().item()})
        if event==256:break
        ids=torch.randint(s,(b,),generator=g);v=torch.randint(0,2,(b,8),generator=g).float()
        table[idx,ids]=v;A=m.write_event(A,ids,v,places[idx,ids])
    assert fingerprint(m)==before
    return {'checkpoint':str(path),'curves':results,'parameters_unchanged':True,'episodes':batch,'queries_per_probe':batch*s}

def main():
    torch.set_num_threads(2);out={}
    for folder in ['results','capacity_results']:
        for path in sorted(Path(folder).glob('*/model.pt')):
            out[str(path.parent)]=evaluate(path)
    Path('stress_results.json').write_text(json.dumps(out,indent=2))
    for k,v in out.items():print(k,[(r['overwrites'],r['exact_accuracy']) for r in v['curves']])
if __name__=='__main__':main()
