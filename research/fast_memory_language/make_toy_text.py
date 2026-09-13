"""Generate original template text for an EXECUTION DEMONSTRATION, not an LM benchmark."""
import itertools,json,random
from pathlib import Path

def main():
    colors=['red','blue','green','gold','silver','white','black','purple']
    actors=['robot','cell','pilot','raven','keeper','farmer','artist','sailor']
    verbs=['carries','finds','stores','moves','hides','paints','counts','inspects']
    things=['key','stone','book','lamp','seed','coin','box','map']
    lines=[f'The {c} {a} {v} the {t}.\n' for c,a,v,t in itertools.product(colors,actors,verbs,things)]
    random.Random(91).shuffle(lines)
    out=Path('toy_text');out.mkdir(exist_ok=True)
    (out/'train.txt').write_text(''.join(lines[:3584]));(out/'validation.txt').write_text(''.join(lines[3584:]))
    (out/'manifest.json').write_text(json.dumps({'status':'Original generated toy grammar; not natural-corpus benchmark',
      'training_sentences':3584,'validation_sentences':512,'sentence_overlap':0,'vocabulary_shared':True},indent=2))
if __name__=='__main__':main()
