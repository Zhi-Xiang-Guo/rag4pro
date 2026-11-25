#!/usr/bin/env python3
"""Bounded local load probe, with 429s counted. This is not a production SLO claim."""
import argparse,concurrent.futures,json,os,pathlib,time,urllib.error,urllib.request
import eval as e

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18080');p.add_argument('--requests',type=int,default=60)
    p.add_argument('--concurrency',default='1,4,16');p.add_argument('--selection',required=True);p.add_argument('--out',default='evaluation/runs/load')
    a=p.parse_args();cfg=e.read(e.ROOT/'configs/retrieval.json');selected=e.read(a.selection)['selected'];qs=e.rows(e.ROOT/'data/scifact/dev.jsonl')[:10]
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);results=[]
    for q in qs:e.request(a.url,'/api/retrieve',dict(cfg['request'],datasetId=cfg['datasetId'],question=q['question'],**cfg['variants'][selected]))
    def call(i):
        body=dict(cfg['request'],datasetId=cfg['datasetId'],question=qs[i%len(qs)]['question'],**cfg['variants'][selected])
        headers={'Content-Type':'application/json'}
        token = os.getenv('EVIDENCE_RAG_HTTP_TOKEN', os.getenv('RAGFLOW4J_HTTP_TOKEN'))
        if token: headers['Authorization'] = 'Bearer ' + token
        req=urllib.request.Request(a.url+'/api/retrieve',data=e.canonical(body),headers=headers);t=time.monotonic()
        try:
            with urllib.request.urlopen(req,timeout=120) as r:r.read();status=r.status;rid=r.headers.get('X-Request-Id')
        except urllib.error.HTTPError as exc:status=exc.code;rid=exc.headers.get('X-Request-Id')
        except Exception:status=0;rid=None
        return {'sequence':i,'status':status,'latencyMs':(time.monotonic()-t)*1000,'requestId':rid}
    summary={'scope':'local Docker, warm query vectors, retrieval only; includes admission failures','variant':selected,'runs':[]}
    for concurrency in map(int,a.concurrency.split(',')):
        t=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:rs=list(pool.map(call,range(a.requests)))
        elapsed=time.monotonic()-t;success=[r for r in rs if r['status']==200]
        for r in rs:r['concurrency']=concurrency
        results.extend(rs)
        summary['runs'].append({'concurrency':concurrency,'requests':len(rs),'successes':len(success),'429':sum(r['status']==429 for r in rs),
            'otherFailures':sum(r['status'] not in (200,429) for r in rs),'wallSeconds':elapsed,'successfulRps':len(success)/elapsed,
            'successP50Ms':e.percentile([r['latencyMs'] for r in success],.5),'successP95Ms':e.percentile([r['latencyMs'] for r in success],.95)})
    e.jsonl(out/'requests.jsonl',results);e.write(out/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
