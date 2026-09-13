"""Sample the byte NCA; supplied trained checkpoint uses artificial template text."""
import argparse
from pathlib import Path
import torch
from causal_lm import CausalNCALM,LMConfig,generate

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--prompt',default='The ');p.add_argument('--bytes',type=int,default=160);p.add_argument('--temperature',type=float,default=.3);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(2);s=torch.load(a.checkpoint,weights_only=True)
    m=CausalNCALM(LMConfig(**s['config']));m.load_state_dict(s['model']);m.eval()
    print(generate(m,a.prompt.encode('utf-8'),a.bytes,a.temperature,a.seed).decode('utf-8',errors='replace'))
if __name__=='__main__':main()
