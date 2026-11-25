#!/usr/bin/env python3
"""Diagnostic generation replay on frozen actual model inputs, with no retrieval/refiner rerun.
This reuses already-seen fixtures; it is not a new held-out result.
"""
import argparse,copy,json,pathlib,random,re,time
import eval as e
from qa import PROMPT

def candidate_messages(messages):
    result=copy.deepcopy(messages)
    marker='      Here is the knowledge base:\n'
    for msg in result:
        if msg['role']=='system':
            prefix,sep,tail=msg['content'].partition(marker)
            if not sep:raise ValueError('unrecognized captured prompt; do not approximate context')
            msg['content']=PROMPT.split('{knowledge}')[0]+sep+tail
            return result,e.digest(tail.encode())
    raise ValueError('no captured system message')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='evaluation/runs/qa-test/results.jsonl')
    p.add_argument('--url',default='http://127.0.0.1:11434');p.add_argument('--out',default='evaluation/runs/replay');a=p.parse_args()
    source=pathlib.Path(a.input);records=[r for r in e.rows(source) if r['variant']=='original']
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    tags=e.request(a.url,'/api/tags');model=next(m for m in tags['models'] if m['name']=='qwen2.5:0.5b')
    meta={'inputSha256':e.digest(source.read_bytes()),'promptSha256':e.digest(PROMPT.encode()),'model':model['name'],
          'modelDigest':model['digest'],'generation':{'temperature':0,'seed':20261124,'num_predict':192,'num_ctx':32768},
          'scope':'diagnostic replay of already-seen synthetic test fixtures; fixed original messages/context, no retrieval or query refinement'}
    if (out/'run.json').exists() and e.read(out/'run.json')!=meta:raise ValueError('replay fingerprint mismatch')
    e.write(out/'run.json',meta)
    results=e.rows(out/'results.jsonl') if (out/'results.jsonl').exists() else []
    done={(r['id'],r['variant']) for r in results}
    for r in records:
        original=r['response'].get('evaluation',{}).get('modelBoundary',{}).get('messages')
        if not original:raise ValueError('original request lacks captured model invocation')
        candidate,tailsha=candidate_messages(original)
        frames=[('original',original),('evidence_first',candidate)]
        random.Random(20261124+sum(r['id'].encode())).shuffle(frames)
        for name,messages in frames:
            if (r['id'],name) in done:continue
            t=time.monotonic();error=None;response={}
            try:response=e.request(a.url,'/api/chat',{'model':meta['model'],'messages':messages,'options':meta['generation'],'stream':False})
            except Exception as exc:error=str(exc)[:500]
            answer=response.get('message',{}).get('content','')
            value={'id':r['id'],'variant':name,'category':r['category'],'unanswerable':r['unanswerable'],
                   'evidenceInFinalContext':r['evidenceInFinalContext'],'contextTailSha256':tailsha,'messages':messages,
                   'rawAnswer':answer,'error':error,'latencyMs':(time.monotonic()-t)*1000,
                   'factPatternPass':not error and bool(re.search(r['answerPattern'],answer,re.I)),
                   'usage':{k:response.get(k) for k in ['prompt_eval_count','eval_count','total_duration']}}
            results.append(value)
            with open(out/'results.jsonl','a') as f:f.write(e.canonical(value).decode()+'\n');f.flush()
            print(r['id'],name,value['factPatternPass'],flush=True)
    summary={'complete':len(results)==len(records)*2,'scope':meta['scope'],'variants':{}}
    for name in ('original','evidence_first'):
        rs=[r for r in results if r['variant']==name];positive=[r for r in rs if not r['unanswerable']];negative=[r for r in rs if r['unanswerable']]
        mean=lambda rows:sum(r['factPatternPass'] for r in rows)/len(rows) if rows else None
        summary['variants'][name]={'queries':len(rs),'errors':sum(bool(r['error']) for r in rs),
              'factPatternPassRate':mean(positive),'refusalPatternPassRate':mean(negative),
              'givenEvidenceFactPatternPassRate':mean([r for r in positive if r['evidenceInFinalContext']])}
    e.write(out/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
