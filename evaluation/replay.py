#!/usr/bin/env python3
"""Diagnostic generation replay on frozen actual model inputs, with no retrieval/refiner rerun.
This reuses already-seen fixtures; it is not a new held-out result.
"""
import argparse,copy,json,os,pathlib,random,re,shlex,time,urllib.request
from urllib.parse import urlsplit
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

def chat_settings():
    values = {}
    env_file = e.ROOT.parent / '.env'
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            name, separator, raw = line.strip().removeprefix('export ').partition('=')
            name = name.strip()
            if not separator or not name.startswith(('EVIDENCE_RAG_', 'RAGFLOW4J_')):
                continue
            fields = shlex.split(raw, comments=True)
            value = fields[0] if len(fields) == 1 else ''
            reference = re.fullmatch(r'\$\{([A-Z_][A-Z0-9_]*)\}', value)
            if reference and not raw.lstrip().startswith("'"):
                value = os.environ.get(reference[1], values.get(reference[1], ''))
            values[name] = value
    values.update(os.environ)

    def get(name, default=''):
        return values.get('EVIDENCE_RAG_' + name, values.get('RAGFLOW4J_' + name)) or default

    return get('CHAT_PROVIDER', 'ollama'), get('CHAT_URL'), get('CHAT_API_KEY'), get('CHAT_MODEL')

def complete_openai(endpoint, api_key, model, messages, generation):
    payload = dict(generation, model=model, messages=messages, stream=False)
    request = urllib.request.Request(endpoint, data=e.canonical(payload),
                                     headers={'Content-Type': 'application/json'})
    request.add_unredirected_header('Authorization', 'Bearer ' + api_key)
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    answer = result['choices'][0]['message']['content']
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('chat model returned no content')
    return answer, result.get('usage') or {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='evaluation/runs/qa-test/results.jsonl')
    p.add_argument('--url', help='Ollama base URL for the historical replay')
    p.add_argument('--out',default='evaluation/runs/replay');a=p.parse_args()
    source=pathlib.Path(a.input);records=[r for r in e.rows(source) if r['variant']=='original']
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    provider, configured_url, api_key, model = chat_settings()
    if a.url:
        provider = 'ollama'
    base = (a.url or configured_url or 'http://127.0.0.1:11434').rstrip('/')
    meta={'inputSha256':e.digest(source.read_bytes()),'promptSha256':e.digest(PROMPT.encode()),
          'scope':'diagnostic replay of already-seen synthetic test fixtures; fixed original messages/context, no retrieval or query refinement'}
    if provider == 'openai':
        url = urlsplit(base)
        if url.scheme != 'https' or not url.hostname or url.username is not None or url.query or url.fragment:
            p.error('EVIDENCE_RAG_CHAT_URL must be an HTTPS base URL')
        if not api_key or not model:
            p.error('EVIDENCE_RAG_CHAT_API_KEY and EVIDENCE_RAG_CHAT_MODEL are required')
        endpoint = base + '/chat/completions'
        meta.update(model=model, modelEndpoint=endpoint,
                    generation={'temperature': 0, 'seed': 20261124, 'max_tokens': 192})
    elif provider == 'ollama':
        tags = e.request(base, '/api/tags')
        historical = next(m for m in tags['models'] if m['name'] == 'qwen2.5:0.5b')
        meta.update(model=historical['name'], modelDigest=historical['digest'],
                    generation={'temperature': 0, 'seed': 20261124, 'num_predict': 192, 'num_ctx': 32768})
    else:
        p.error('EVIDENCE_RAG_CHAT_PROVIDER must be ollama or openai')
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
            t = time.monotonic()
            error, answer, usage = None, '', {}
            try:
                if provider == 'openai':
                    answer, usage = complete_openai(endpoint, api_key, model, messages, meta['generation'])
                else:
                    response = e.request(base, '/api/chat', {'model': meta['model'], 'messages': messages,
                                                           'options': meta['generation'], 'stream': False})
                    answer = response.get('message', {}).get('content', '')
                    usage = {key: response.get(key) for key in ['prompt_eval_count', 'eval_count', 'total_duration']}
            except Exception as exc:
                error = str(exc)[:500]
            value={'id':r['id'],'variant':name,'category':r['category'],'unanswerable':r['unanswerable'],
                   'evidenceInFinalContext':r['evidenceInFinalContext'],'contextTailSha256':tailsha,'messages':messages,
                   'rawAnswer':answer,'error':error,'latencyMs':(time.monotonic()-t)*1000,
                   'factPatternPass':not error and bool(re.search(r['answerPattern'],answer,re.I)),
                   'usage':usage}
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
