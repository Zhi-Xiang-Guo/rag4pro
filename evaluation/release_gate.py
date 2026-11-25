#!/usr/bin/env python3
"""A failed generation/safety check cannot be hidden by a successful retrieval benchmark."""
import argparse,csv,json,pathlib,sys
import eval as e

def evaluate(retrieval, qa, load, reviewed=False, selected=None):
    failures=[]
    if not retrieval.get('passed'):failures.append('retrieval quality gate failed')
    if not qa.get('complete'):failures.append('QA run is incomplete')
    for name,v in qa.get('variants',{}).items():
        if selected is not None and name!=selected:continue
        if v['errors']>0:failures.append(name+': upstream errors observed')
        if v['factPatternPassRate']<.9:failures.append(name+': factual regression checks below 90%')
        if v['refusalPatternPassRate']<1:failures.append(name+': unsupported-question refusal checks failed')
        if name == 'extractive':
            if v.get('strictRefusalPassRate') != 1:
                failures.append('extractive: unsupported questions must return an explicit refusal without citations')
            if v.get('verifiedSpanCoverage') != 1:
                failures.append('extractive: every answer must be rendered from verified source spans')
            if (v.get('factsWithExpectedEvidenceRate') or 0) < .9:
                failures.append('extractive: factual checks with expected evidence below 90%')
    if selected is not None and selected not in qa.get('variants',{}):failures.append('missing selected QA variant')
    if any(r['otherFailures'] for r in load.get('runs',[])):failures.append('load probe has non-admission errors')
    if not load.get('runs'):failures.append('missing bounded load probe')
    automated_passed = not failures
    if not reviewed:failures.append('independent human correctness/faithfulness/citation review pending')
    # Other production readiness obligations are deliberately not inferred from these offline gates.
    return {'passed':not failures,'automatedPassed':automated_passed,
            'failures':failures,'scope':'evaluation acceptance only; not a production certification',
            'policy':{'factPatternPassRate':.9,'refusalPatternPassRate':1,'errors':0,'humanReviewRequired':True},
            'productionReadiness':'not certified'}

def validate_review(path, records, selected):
    expected={(r['id'],r['variant']):r for r in records if r['variant']==selected}
    if not path:return {'passed':False,'reason':'no review file supplied'}
    with open(path) as f: rows=list(csv.DictReader(f))
    actual={};scores={'correctness':[],'faithfulness':[],'citationSupport':[]}
    for row in rows:
        if row['variant']!=selected:continue
        key=(row['id'],row['variant'])
        if key not in expected or key in actual: return {'passed':False,'reason':'unknown or duplicate review key'}
        r=expected[key]
        if row['question']!=r['question'] or row['answer']!=r['response'].get('answer',''):
            return {'passed':False,'reason':'review does not match captured question/answer'}
        if not row['reviewer'].strip():return {'passed':False,'reason':'reviewer missing'}
        for field in scores:
            value=row[field+'_0_1']
            if field=='citationSupport' and r['unanswerable'] and value=='NA':continue
            if value not in ('0','1'):return {'passed':False,'reason':'incomplete binary review scores'}
            scores[field].append(int(value))
        actual[key]=row
    if set(actual)!=set(expected):return {'passed':False,'reason':'incomplete selected-variant review coverage'}
    means={k:sum(v)/len(v) if v else 1 for k,v in scores.items()}
    passed=means['correctness']>=.9 and means['faithfulness']>=.95 and means['citationSupport']>=.95
    return {'passed':passed,'means':means,'rows':len(actual),'sha256':e.digest(pathlib.Path(path).read_bytes()),
            'method':'user-supplied declared human review; independent identity not automatically verified'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--retrieval',default='evaluation/runs/test/gate.json')
    p.add_argument('--qa',default='evaluation/runs/qa-test/summary.json');p.add_argument('--load',default='evaluation/runs/load/summary.json')
    p.add_argument('--out',default='evaluation/runs/release-gate.json');p.add_argument('--review')
    p.add_argument('--qa-selection',default='evaluation/runs/qa-dev/selection.json');a=p.parse_args()
    selected=e.read(a.qa_selection)['selected']
    records=e.rows(pathlib.Path(a.qa).parent/'results.jsonl')
    review=validate_review(a.review,records,selected)
    result=evaluate(e.read(a.retrieval),e.read(a.qa),e.read(a.load),reviewed=review['passed'],selected=selected)
    result['qaSelected']=selected;result['humanReview']=review;e.write(a.out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':sys.exit(main())
