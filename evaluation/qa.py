#!/usr/bin/env python3
"""Local-model answer/citation regression. Deterministic checks are proxies, not faithfulness judgments."""
import argparse,csv,json,pathlib,re,sys,time,random
import eval as e
from qa_checks import explicit_refusal, extractive_context, source_text, verified_spans
ROOT=pathlib.Path(__file__).resolve().parent
PROMPT='''你是知识库问答助手。仅根据下方证据回答用户最后的问题，结合对话确定问题中的产品和版本。
先确认产品、版本及问题所问的字段；不同产品或旧版本的数据不能混用。
证据没有直接给出答案时，只回答“资料不足，无法回答”，不要推测。回答简洁，保留数字和单位，并引用证据 ID。
知识库：\n{knowledge}'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18080');p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--out',default=str(ROOT/'runs/qa-dev'));p.add_argument('--selection')
    p.add_argument('--candidate', choices=['evidence_first', 'extractive'], default='evidence_first')
    a=p.parse_args()
    data=e.read(ROOT/'qa-fixture.json');out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfgsha=e.digest(e.canonical({'fixture':data,'prompt':PROMPT}))
    health = e.request(a.url, '/api/health')
    variants = ['original', 'evidence_first']
    model_keys = ['embeddingIdentity', 'chatModel', 'chatContext']
    if a.candidate == 'extractive':
        variants = ['extractive']
        model_keys.extend(['extractiveModel', 'extractiveContext'])
        cfgsha = e.digest(e.canonical({'baselineConfig': cfgsha, 'candidate': 'extractive',
                                     'runtime': {key: health[key] for key in model_keys},
                                     'source': e.source_fingerprint(),
                                     'runner': e.digest(pathlib.Path(__file__).read_bytes()),
                                     'checks': e.digest((ROOT/'qa_checks.py').read_bytes())}))
        corpus_hash = e.digest(e.canonical(data['documents']))
        data['datasetId'] = f"eval-controlled-qa-v2-{corpus_hash[:12]}"
        for document in data['documents']:
            document.setdefault('metadata', {})['qa_corpus_sha256'] = corpus_hash
    if a.split=='test':
        if not a.selection: raise ValueError('freeze dev prompt selection before test')
        selected=e.read(a.selection)
        if selected['configSha256']!=cfgsha: raise ValueError('QA configuration drift')
        if a.candidate == 'evidence_first':
            variants=list(dict.fromkeys(['original',selected['selected']]))
        elif selected['selected'] != 'extractive':
            raise ValueError('test requires frozen extractive candidate')
    docs=e.request(a.url,'/api/documents?datasetId='+data['datasetId']);done={d['docId'] for d in docs}
    if a.candidate == 'extractive' and any(
            d.get('metadata', {}).get('qa_corpus_sha256') != corpus_hash for d in docs):
        raise ValueError('QA corpus version drift')
    missing=[d for d in data['documents'] if d['docId'] not in done]
    if missing:e.request(a.url,'/api/documents/batch',{'datasetId':data['datasetId'],'documents':missing})
    queries=[q for q in data['queries'] if q['split']==a.split]
    meta={'split':a.split,'configSha256':cfgsha,'labelStatus':data['labelStatus'],'variants':variants,
          'queryIds':[q['id'] for q in queries],'health':health,'datasetId':data['datasetId']}
    if (out/'run.json').exists():
        old=e.read(out/'run.json')
        for key in ('split','configSha256','variants','queryIds'):
            if old[key]!=meta[key]: raise ValueError('run fingerprint drift: '+key)
        for key in model_keys:
            if old['health'][key]!=meta['health'][key]:raise ValueError('runtime model drift: '+key)
    else:e.write(out/'run.json',meta)
    records=e.rows(out/'results.jsonl') if (out/'results.jsonl').exists() else []
    done={(r['id'],r['variant']) for r in records}
    for q in queries:
        order=list(variants); random.Random(20261124+sum(q['id'].encode())).shuffle(order)
        for variant in order:
            if (q['id'],variant) in done:continue
            req={'datasetId':data['datasetId'],'question':q['question'],'topK':4,'threshold':.1,
                 'generation':{'temperature':0,'max_tokens':192,'seed':20261124},'evaluationTrace':True}
            for key in ('messages','metadataFilter'):
                if key in q:req[key]=q[key]
            if variant=='evidence_first':req['systemPrompt']=PROMPT
            if variant == 'extractive':
                req['answerMode'] = 'extractive'
                req['strategy'] = 'rrf'
            t=time.monotonic();error=None;response={}
            try:response=e.request(a.url,'/api/chat',req)
            except Exception as exc:error=str(exc)[:500]
            answer=response.get('answer',''); chunks=response.get('reference',{}).get('chunks',[])
            boundary=response.get('evaluation',{}).get('modelBoundary',{})
            marks=[int(m) for m in re.findall(r'\[(?:ID:\s*)?(\d+)\]',answer)]
            evidence=q.get('evidence',''); final_messages=boundary.get('messages',[])
            context = (extractive_context(boundary) if variant == 'extractive' else
                       '\n'.join(m['content'] for m in final_messages if m['role']=='system'))
            fact_match=not error and bool(re.search(q['answerPattern'],answer,re.I))
            valid=bool(marks) and all(0<=n<len(chunks) for n in marks)
            # Neither marker validity nor string overlap proves entailment/faithfulness.
            r={'id':q['id'],'variant':variant,'category':q['category'],'unanswerable':q.get('unanswerable',False),
               'question':q['question'],'expectedEvidence':evidence,'answerPattern':q['answerPattern'],
               'latencyMs':(time.monotonic()-t)*1000,'error':error,'factPatternPass':fact_match,
               'evidenceInFinalContext':bool(evidence) and evidence in context,
               'citationSyntaxValid':valid,
               'manualFaithfulness':'pending','response':response}
            with open(out/'results.jsonl','a') as f:f.write(e.canonical(r).decode()+'\n');f.flush()
            records.append(r);print(q['id'],variant,fact_match,flush=True)
    for r in records:
        chunks=r['response'].get('reference',{}).get('chunks',[]); evidence=r['expectedEvidence']
        marks=[int(m) for m in re.findall(r'\[(?:ID:\s*)?(\d+)\]',r['response'].get('answer',''))]
        retrieved = r['response'].get('evaluation', {}).get('retrievedChunks', chunks)
        r['evidenceRetrieved'] = bool(evidence) and any(evidence in source_text(c) for c in retrieved)
        if r['variant'] == 'extractive':
            spans = verified_spans(r['response'])
            r['exactSpanVerification'] = bool(spans)
            r['strictRefusalPass'] = explicit_refusal(r['response'])
            r['expectedEvidenceCited'] = bool(evidence) and any(evidence in s['text'] for s in spans)
        else:
            r['expectedEvidenceCited'] = (bool(evidence) and r['citationSyntaxValid']
                                          and any(evidence in source_text(chunks[n]) for n in marks))
    e.jsonl(out/'results.jsonl',records)
    summary={'split':a.split,'labelStatus':data['labelStatus'],'metricsMeaning':'Rule/regex regression checks; not semantic accuracy or faithfulness. Latency includes mixed cache states and is observational.',
             'complete':len(records)==len(queries)*len(variants) and
                        {(r['id'], r['variant']) for r in records} ==
                        {(q['id'], v) for q in queries for v in variants},'variants':{}}
    for variant in variants:
        rs=[r for r in records if r['variant']==variant];answerable=[r for r in rs if not r['unanswerable']];negative=[r for r in rs if r['unanswerable']]
        mean=lambda key,items:sum(bool(r[key]) for r in items)/len(items) if items else None
        summary['variants'][variant]={'queries':len(rs),'errors':sum(bool(r['error']) for r in rs),
            'factPatternPassRate':mean('factPatternPass',answerable),'refusalPatternPassRate':mean('factPatternPass',negative),
            'evidenceRecall':mean('evidenceRetrieved',answerable),'finalContextEvidenceRate':mean('evidenceInFinalContext',answerable),
            'citationSyntaxValidRate':mean('citationSyntaxValid',answerable),'expectedEvidenceCitedRate':mean('expectedEvidenceCited',answerable),
            'latencyP95Ms':e.percentile([r['latencyMs'] for r in rs],.95),'manualFaithfulness':'pending'}
        if variant == 'extractive':
            answered = [r for r in rs if r['response'].get('answerStatus') == 'answered']
            summary['variants'][variant].update({
                'answered': len(answered),
                'strictRefusalPassRate': mean('strictRefusalPass', negative),
                'verifiedSpanCoverage': mean('exactSpanVerification', answered),
                'factsWithExpectedEvidenceRate': sum(r['factPatternPass'] and r['expectedEvidenceCited']
                                                     for r in answerable) / len(answerable),
            })
    e.write(out/'summary.json',summary)
    with open(out/'human-review.csv','w') as f:
        w=csv.writer(f);w.writerow(['id','variant','question','answer','goldEvidence','correctness_0_1','faithfulness_0_1','citationSupport_0_1','reviewer','note'])
        for r in records:w.writerow([r['id'],r['variant'],r['question'],r['response'].get('answer',''),r['expectedEvidence'],'','','','',''])
    if a.split == 'dev' and summary['complete'] and a.candidate == 'extractive':
        e.write(out/'selection.json', {'selected': 'extractive', 'configSha256': cfgsha,
                                      'devResultsSha256': e.digest((out/'results.jsonl').read_bytes())})
    elif a.split=='dev' and summary['complete'] and all(v['errors']==0 for v in summary['variants'].values()):
        # Select using only dev rule pass across all categories. Original wins ties to avoid unsupported upgrade claims.
        score=lambda v:sum(r['factPatternPass'] for r in records if r['variant']==v)
        chosen='evidence_first' if score('evidence_first')>score('original') else 'original'
        e.write(out/'selection.json',{'selected':chosen,'configSha256':cfgsha,'devResultsSha256':e.digest((out/'results.jsonl').read_bytes())})
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
