"""Acquire fresh random associations with frozen parameters, then query remotely."""
import argparse,json
from pathlib import Path
import torch
from cellular_memory import CellularMemory,MemoryConfig
from tasks import sample,ingest
from run_memory import fingerprint,score

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--seed',type=int,default=92312);p.add_argument('--records',type=int,default=8);p.add_argument('--cells',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(2)
    saved=torch.load(a.checkpoint,weights_only=True);m=CellularMemory(MemoryConfig(**saved['config']));m.load_state_dict(saved['model']);m.eval()
    e=sample(8,a.cells,a.records,torch.Generator().manual_seed(a.seed));before=fingerprint(m)
    A=ingest(m,e);z=m.query(A,e.query_symbol,e.query_origin)
    def bits(x):return ''.join(str(int(v)) for v in x)
    rows=[{'key':int(e.query_symbol[i]),'writer':int(e.source_position[i]),'reader':int(e.query_origin[i]),
           'observed':bits(e.target[i]),'answer':bits(z[i]>0)} for i in range(len(z))]
    print(json.dumps({'examples':rows,'scores':score(z,e.target),'shared_weights_unchanged':fingerprint(m)==before},indent=2))
if __name__=='__main__':main()
