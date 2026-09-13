"""Byte-level NCA trainer for TWO explicit, separate local text files.

Loss at input position i predicts byte i+1. No future-dependent adaptation is
performed. A byte vocabulary makes nats/log(2) a genuine bits-per-byte measure.
This example has a finite receptive field, not unlimited recurrent context.
"""
from __future__ import annotations
import argparse,json,math,hashlib,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from causal_lm import CausalNCALM,LMConfig,generate

def make_batch(data:torch.Tensor,batch:int,length:int,g:torch.Generator,device):
    if len(data)<=length:raise ValueError('Data must be longer than sequence length')
    start=torch.randint(len(data)-length,(batch,),generator=g)
    block=data[start[:,None]+torch.arange(length+1)[None,:]].to(device)
    return block[:,:-1],block[:,1:]

@torch.no_grad()
def evaluate(model,data,length,batches=16):
    model.eval();g=torch.Generator().manual_seed(22002);losses=[]
    for _ in range(batches):
        x,y=make_batch(data,8,length,g,model.embedding.weight.device)
        losses.append(F.cross_entropy(model(x).reshape(-1,256),y.reshape(-1)).item())
    nats=sum(losses)/len(losses)
    return {'sampled_validation_nats_per_byte':nats,'sampled_validation_bits_per_byte':nats/math.log(2),
            'note':'Mean over fixed sampled validation windows, including left-edge warmup positions'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--length',type=int,default=128);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--channels',type=int,default=32);p.add_argument('--hidden',type=int,default=64);p.add_argument('--iterations',type=int,default=8)
    p.add_argument('--radius',type=int,default=2);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cpu');p.add_argument('--out',type=Path,default=Path('language_run'))
    a=p.parse_args()
    if min(a.steps,a.length,a.batch)<1:p.error('steps, length and batch must be positive')
    train_bytes=a.train.read_bytes();val_bytes=a.validation.read_bytes()
    if a.train.resolve()==a.validation.resolve() or train_bytes==val_bytes:p.error('Training and validation must differ')
    if min(len(train_bytes),len(val_bytes))<=a.length:p.error('Files must exceed the sequence length')
    torch.set_num_threads(2);torch.manual_seed(a.seed);torch.use_deterministic_algorithms(True)
    m=CausalNCALM(LMConfig(channels=a.channels,hidden=a.hidden,radius=a.radius,iterations=a.iterations)).to(a.device)
    tr=torch.tensor(list(train_bytes),dtype=torch.long);va=torch.tensor(list(val_bytes),dtype=torch.long)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.01);g=torch.Generator().manual_seed(1000+a.seed)
    a.out.mkdir(parents=True,exist_ok=True);before=evaluate(m,va,a.length);start=time.perf_counter()
    with (a.out/'training.jsonl').open('w') as f:
        for step in range(1,a.steps+1):
            m.train();x,y=make_batch(tr,a.batch,a.length,g,a.device)
            opt.zero_grad(set_to_none=True);logits=m(x)
            loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            row={'step':step,'nats_per_byte':loss.item()};f.write(json.dumps(row)+'\n')
            if step%100==0 or step==1:print(row,flush=True)
    result={'config':m.config_dict(),'parameters':m.parameter_count(),'training_steps':a.steps,
            'processed_byte_targets':a.steps*a.batch*a.length,'max_context_bytes':m.max_context_bytes(),
            'before':before,'after':evaluate(m,va,a.length),'seconds':time.perf_counter()-start,
            'train_sha256':hashlib.sha256(train_bytes).hexdigest(),'validation_sha256':hashlib.sha256(val_bytes).hexdigest(),
            'torch':str(torch.__version__),'device':a.device,
            'limitations':'No fast-memory integration; fixed local context; no general language claim from this run'}
    torch.save({'model':m.state_dict(),'config':m.config_dict()},a.out/'model.pt')
    result['sample']=generate(m,b'The ',120,seed=a.seed).decode('utf-8',errors='replace')
    (a.out/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
