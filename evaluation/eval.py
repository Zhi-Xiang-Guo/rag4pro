#!/usr/bin/env python3
"""Reproducible retrieval evaluation. Labels never travel to the application.
No third-party dependency except pyarrow for downloading/converting parquet.
"""
import argparse, collections, concurrent.futures, csv, datetime, hashlib, json, math, os, pathlib, random, statistics, sys, time, urllib.request, urllib.error
ROOT = pathlib.Path(__file__).resolve().parent

def digest(data): return hashlib.sha256(data).hexdigest()
def canonical(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
def read(path): return json.loads(pathlib.Path(path).read_text())
def write(path, obj):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp'); tmp.write_bytes(json.dumps(obj, ensure_ascii=False, indent=2).encode()); tmp.replace(path)
def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
def jsonl(path, values):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for obj in values: f.write(canonical(obj).decode()+'\n')
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'evidence-rag-eval/1'}), timeout=120) as r: return r.read()
def request(base, path, body=None, timeout=300, method=None):
    headers = {'Content-Type':'application/json'}
    token = os.getenv('EVIDENCE_RAG_HTTP_TOKEN', os.getenv('RAGFLOW4J_HTTP_TOKEN'))
    if token: headers['Authorization'] = 'Bearer ' + token
    req = urllib.request.Request(base+path, data=None if body is None else canonical(body), headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)
    except urllib.error.HTTPError as exc:
        body=exc.read().decode(errors='replace')[:1000]
        raise RuntimeError(f'HTTP {exc.code}: {body}') from None

def prepare(args):
    target=pathlib.Path(args.data)
    if (target/'manifest.json').exists():
        verified_data(args); print('Existing frozen dataset verified'); return
    import pyarrow.parquet as pq
    from io import BytesIO, StringIO
    target=pathlib.Path(args.data); target.mkdir(parents=True,exist_ok=True)
    sources=[]
    def pinned(repo, filename):
        info=json.loads(fetch('https://huggingface.co/api/datasets/'+repo))
        rev=info['sha']; url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{filename}'
        data=fetch(url); sources.append({'url':url,'sha256':digest(data),'bytes':len(data),'repository':repo,'revision':rev})
        return data
    corpus=pq.read_table(BytesIO(pinned('BeIR/scifact','corpus/corpus-00000-of-00001.parquet'))).to_pylist()
    queries=pq.read_table(BytesIO(pinned('BeIR/scifact','queries/queries-00000-of-00001.parquet'))).to_pylist()
    query_map={str(q['_id']):q['text'] for q in queries}
    splits={}
    for split in ('train','test'):
        qrels=collections.defaultdict(dict)
        for row in csv.DictReader(StringIO(pinned('BeIR/scifact-qrels',split+'.tsv').decode()),delimiter='\t'):
            qrels[str(row['query-id'])][str(row['corpus-id'])]=int(row['score'])
        ids=sorted(qrels, key=lambda q: digest(('ragflow4j-dev-v1:'+q).encode()))
        # Official train is development only; test is never used for selection.
        if split=='train': ids=ids[:60]
        splits['dev' if split=='train' else 'test']=[{'id':q,'question':query_map[q],'qrels':qrels[q]} for q in ids]
    assert len(corpus)==5183 and len(splits['test'])==300
    ids={str(d['_id']) for d in corpus}
    assert all(set(q['qrels'])<=ids for qs in splits.values() for q in qs)
    assert not ({q['id'] for q in splits['dev']} & {q['id'] for q in splits['test']})
    jsonl(target/'corpus.jsonl',[{'id':str(d['_id']),'title':d['title'],'text':d['text']} for d in corpus])
    for split,qs in splits.items(): jsonl(target/(split+'.jsonl'),qs)
    manifest={'schemaVersion':1,'name':'BEIR SciFact full corpus','license':'CC-BY-SA-4.0',
              'licenseSource':'https://huggingface.co/datasets/BeIR/scifact',
              'methodSource':'https://github.com/beir-cellar/beir/wiki/Datasets-available',
              'createdUtc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'corpusCount':len(corpus),'splits':{k:len(v) for k,v in splits.items()},'sources':sources,
              'files':{p.name:digest(p.read_bytes()) for p in target.glob('*.jsonl')},
              'limitations':['Scientific claim to document retrieval; no answer or evidence-span gold labels.',
                              'Official split may share topics/near-duplicate claims. Not domain generalization or QA accuracy.',
                              'Unjudged documents treated as irrelevant; qrels are not exhaustive.']}
    write(target/'manifest.json',manifest); print(json.dumps(manifest,indent=2))

def verified_data(args):
    data=pathlib.Path(args.data); manifest=read(data/'manifest.json')
    for name,sha in manifest['files'].items():
        if digest((data/name).read_bytes())!=sha: raise ValueError('dataset drift: '+name)
    return data,manifest

def ingest(args):
    data,manifest=verified_data(args); cfg=read(args.config)
    if not 1 <= args.batch <= 64: raise ValueError('batch must be 1..64')
    dataset=cfg['datasetId']; done={d['docId'] for d in request(args.url,'/api/documents?datasetId='+dataset)}
    present=request(args.url,'/api/documents?datasetId='+dataset)
    if any(d.get('metadata',{}).get('corpusSha256')!=manifest['files']['corpus.jsonl'] for d in present): raise ValueError('resume index provenance mismatch')
    corpus=rows(data/'corpus.jsonl'); pending=[d for d in corpus if d['id'] not in done]
    # Provenance in metadata is data-only; no relevance labels/questions are sent.
    for start in range(0,len(pending),args.batch):
        batch=pending[start:start+args.batch]; t=time.monotonic()
        body={'datasetId':dataset,'documents':[{'docId':d['id'],'title':d['title'],'text':d['text'],
            'language':'English','fileType':'txt','chunkTokenSize':512,'filenameEmbeddingWeight':0.1,
            'metadata':{'source':'BEIR SciFact','corpusSha256':manifest['files']['corpus.jsonl']}} for d in batch]}
        result=request(args.url,'/api/documents/batch',body,timeout=900)
        print(json.dumps({'indexed':len(done)+start+len(batch),'of':len(corpus),'seconds':round(time.monotonic()-t,2),'result':result}),flush=True)
    actual=request(args.url,'/api/documents?datasetId='+dataset)
    if {d['docId'] for d in actual}!={d['id'] for d in corpus}: raise ValueError('corpus/catalog mismatch')
    if any(d.get('metadata',{}).get('corpusSha256')!=manifest['files']['corpus.jsonl'] for d in actual): raise ValueError('corpus provenance mismatch')
    write(data/'indexed.json',{'datasetId':dataset,'documents':len(actual),'manifestSha256':digest(canonical(manifest))})

def metrics(ranking,qrels):
    # Collapse duplicate documents BEFORE applying k, preserving the highest-ranked chunk.
    ranked=list(dict.fromkeys(ranking)); positives={d for d,v in qrels.items() if v>0}
    if not positives: raise ValueError('query has no positive judgments')
    result={}
    for k in (1,5,10):
        rel=[qrels.get(d,0) for d in ranked[:k]]
        result[f'recall@{k}']=sum(v>0 for v in rel)/len(positives)
        result[f'precision@{k}']=sum(v>0 for v in rel)/k
        result[f'mrr@{k}']=next((1/(i+1) for i,v in enumerate(rel) if v>0),0)
        dcg=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(rel))
        ideal=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(sorted(qrels.values(),reverse=True)[:k]))
        result[f'ndcg@{k}']=dcg/ideal
    return result

def source_fingerprint():
    base=ROOT.parent
    files=[base/'pom.xml']+sorted((base/'src/main').rglob('*'))+sorted((base/'evaluation/configs').glob('*.json'))+[pathlib.Path(__file__)]
    return digest(b''.join(str(p.relative_to(base)).encode()+b'\0'+p.read_bytes() for p in files if p.is_file()))

def run(args):
    data,manifest=verified_data(args); cfg=read(args.config); out=pathlib.Path(args.out); out.mkdir(parents=True,exist_ok=True)
    variants=cfg['variants']
    selection=None
    if args.split=='test':
        if not args.selection: raise ValueError('test requires frozen dev --selection')
        selection=read(args.selection)
        if selection['configSha256']!=digest(canonical(cfg)) or selection['dataSha256']!=digest(canonical(manifest)): raise ValueError('selection/data/config mismatch')
        variants={k:v for k,v in variants.items() if k in (cfg['baseline'],selection['selected'])}
    if selection and selection['sourceSha256'] != source_fingerprint(): raise ValueError('source changed since dev selection')
    health=request(args.url,'/api/health')
    docs=request(args.url,'/api/documents?datasetId='+cfg['datasetId'])
    corpus=rows(data/'corpus.jsonl')
    if {d['docId'] for d in docs}!={d['id'] for d in corpus}: raise ValueError('full corpus not indexed')
    if any(d.get('metadata',{}).get('corpusSha256')!=manifest['files']['corpus.jsonl'] for d in docs): raise ValueError('index provenance mismatch')
    queries=rows(data/(args.split+'.jsonl'))
    meta={'queryIds':[q['id'] for q in queries],'schemaVersion':1,'split':args.split,'config':cfg,'configSha256':digest(canonical(cfg)),
          'dataSha256':digest(canonical(manifest)),'sourceSha256':source_fingerprint(),
          'baseUrl':args.url,'health':health,'queryCache':'warm query vectors after explicit per-query warmup' if args.warmup else 'no explicit warmup; see server env',
          'variants':list(variants),'selection':selection,'startedUtc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    meta_path=out/'run.json'
    if meta_path.exists():
        old=read(meta_path)
        for key in ('split','configSha256','dataSha256','sourceSha256','baseUrl','variants','queryCache','queryIds'):
            if old[key]!=meta[key]: raise ValueError('resume fingerprint differs: '+key)
    else: write(meta_path,meta)
    artifact=out/'results.jsonl'; existing=rows(artifact) if artifact.exists() else []
    done={(r['queryId'],r['variant']) for r in existing}
    queries=rows(data/(args.split+'.jsonl'))
    for qi,q in enumerate(queries):
        order=list(variants); random.Random(cfg['seed']+qi).shuffle(order)
        if args.warmup and any((q['id'],v) not in done for v in order):
            request(args.url,'/api/retrieve',dict(cfg['request'],datasetId=cfg['datasetId'],question=q['question'],strategy='dense',topK=1))
        for variant in order:
            if (q['id'],variant) in done: continue
            body=dict(cfg['request'],**variants[variant],datasetId=cfg['datasetId'],question=q['question'])
            t=time.monotonic(); error=None; result={}
            try: result=request(args.url,'/api/retrieve',body)
            except Exception as exc: error=type(exc).__name__+': '+str(exc)[:200]
            elapsed=(time.monotonic()-t)*1000
            ranking=[c['doc_id'] for c in result.get('chunks',[])]
            # Internal IDs are UUID v3 MD5(dataset NUL docId), as DocumentCatalog.identity.
            import uuid
            mapping={uuid.UUID(bytes=hashlib.md5((cfg['datasetId']+'\0'+d['id']).encode()).digest(),version=3).hex:d['id'] for d in corpus}
            external=[mapping.get(d,'UNKNOWN:'+d) for d in ranking]
            record={'queryId':q['id'],'variant':variant,'question':q['question'],'latencyMs':elapsed,'error':error,
                'ranking':list(dict.fromkeys(external)),'returnedChunks':len(ranking),'metrics':metrics(external,q['qrels']),
                'chunks':[{'id':c.get('chunk_id'),'docId':mapping.get(c.get('doc_id')), 'score':c.get('similarity'),
                           'text':c.get('content_with_weight','')} for c in result.get('chunks',[])],
                'stages':[{k:v for k,v in stage.items() if k!='request'} for stage in result.get('trace',[])]}
            with open(artifact,'a') as f: f.write(canonical(record).decode()+'\n'); f.flush(); os.fsync(f.fileno())
            done.add((q['id'],variant))
        print(f'{args.split} {qi+1}/{len(queries)}',flush=True)
    report_dir(out, expected=len(queries)*len(variants))

def percentile(xs,p):
    xs=sorted(xs)
    return xs[max(0,math.ceil(p*len(xs))-1)] if xs else None

def report_dir(out,expected=None):
    out=pathlib.Path(out); records=rows(out/'results.jsonl'); meta=read(out/'run.json')
    keys=[(r['queryId'],r['variant']) for r in records]
    if len(keys)!=len(set(keys)): raise ValueError('duplicate result keys')
    groups=collections.defaultdict(list)
    for r in records: groups[r['variant']].append(r)
    expected_keys={(q,v) for q in meta['queryIds'] for v in meta['variants']}
    summary={'split':meta['split'],'complete':set(keys)==expected_keys,'variants':{}}
    for name,rs in groups.items():
        summary['variants'][name]={'queries':len(rs),'errors':sum(r['error'] is not None for r in rs),
            'errorRate':sum(r['error'] is not None for r in rs)/len(rs),
            'latencyP50Ms':percentile([r['latencyMs'] for r in rs],.5),'latencyP95Ms':percentile([r['latencyMs'] for r in rs],.95),
            **{key:statistics.mean(r['metrics'][key] for r in rs) for key in rs[0]['metrics']}}
    baseline=meta['config']['baseline']; metric=meta['config']['primaryMetric']; summary['comparisons']={}
    if baseline not in groups: raise ValueError('missing baseline results')
    base={r['queryId']:r for r in groups[baseline]}
    for name,rs in groups.items():
        if name==baseline: continue
        if set(base)!={r['queryId'] for r in rs}: raise ValueError('unpaired comparison')
        delta=[r['metrics'][metric]-base[r['queryId']]['metrics'][metric] for r in rs]
        rng=random.Random(meta['config']['seed']); samples=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(2000))
        summary['comparisons'][name]={'primaryMetric':metric,'absoluteDelta':statistics.mean(delta),
                'pairedBootstrap95CI':[samples[49],samples[1949]],'resamples':2000,
                'wins':sum(x>0 for x in delta),'ties':sum(x==0 for x in delta),'losses':sum(x<0 for x in delta),
                'limitation':'Query bootstrap assumes independent claims; shared topics may narrow intervals.'}
    write(out/'summary.json',summary)
    cols=['variant']+list(next(iter(summary['variants'].values())).keys())
    with open(out/'summary.csv','w') as f:
        w=csv.DictWriter(f,cols);w.writeheader()
        for k,v in summary['variants'].items(): w.writerow(dict(variant=k,**v))
    # Both improvements and regressions, including full evidence, are available for diagnosis.
    bad=[]
    for name,rs in groups.items():
        if name==baseline: continue
        diffs=sorted(rs,key=lambda r:r['metrics'][metric]-base[r['queryId']]['metrics'][metric])
        for r in diffs[:10]+diffs[-10:]: bad.append({'variant':name,'baseline':base[r['queryId']],'candidate':r,'delta':r['metrics'][metric]-base[r['queryId']]['metrics'][metric]})
    jsonl(out/'cases.jsonl',bad)
    print(json.dumps(summary,indent=2)); return summary

def select(args):
    out=pathlib.Path(args.out); meta=read(out/'run.json'); summary=read(out/'summary.json'); cfg=meta['config']
    if meta['split']!='dev' or not summary['complete']: raise ValueError('selection requires complete dev')
    metric=cfg['primaryMetric']; chosen=max(cfg['selectionCandidates'],key=lambda k:(summary['variants'][k][metric],k))
    if any(v['errorRate']>0 for v in summary['variants'].values()): raise ValueError('resolve dev errors before selecting')
    write(out/'selection.json',{'selected':chosen,'metric':metric,'reason':'highest dev metric among predeclared candidates',
        'configSha256':meta['configSha256'],'dataSha256':meta['dataSha256'],'sourceSha256':meta['sourceSha256'],
        'devResultsSha256':digest((out/'results.jsonl').read_bytes())})
    print(chosen)

def gate(args):
    out=pathlib.Path(args.out); meta=read(out/'run.json'); s=read(out/'summary.json'); cfg=meta['config']; issues=[]
    if meta['split']!='test' or not s['complete']: issues.append('requires complete held-out test')
    chosen=meta.get('selection',{}).get('selected'); limits=cfg['gates']
    if not chosen or chosen not in s['variants']: issues.append('missing frozen candidate')
    else:
        v=s['variants'][chosen]
        if v['errorRate']>limits['maxErrorRate']: issues.append('error rate gate')
        if v['recall@10']<limits['minRecall@10']: issues.append('recall gate')
        if s['comparisons'][chosen]['absoluteDelta']<limits['minPrimaryDelta']: issues.append('quality regression gate')
    write(out/'gate.json',{'passed':not issues,'failures':issues,'limits':limits,'scope':'retrieval only; does not certify production readiness'})
    print('PASS' if not issues else 'FAIL: '+', '.join(issues)); return 1 if issues else 0

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','ingest','run','report','select','gate'])
    p.add_argument('--data',default=str(ROOT/'data/scifact'));p.add_argument('--config',default=str(ROOT/'configs/retrieval.json'))
    p.add_argument('--url',default='http://127.0.0.1:18080');p.add_argument('--out',default=str(ROOT/'runs/dev'))
    p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--selection');p.add_argument('--batch',type=int,default=32);p.add_argument('--warmup',action='store_true')
    a=p.parse_args()
    if a.command=='report': report_dir(a.out)
    else: return globals()[a.command](a) or 0
if __name__=='__main__': sys.exit(main())
