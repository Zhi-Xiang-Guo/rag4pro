import { useState, type FormEvent } from 'react';
import { messageOf, type ApiClient, type Dataset } from './api';
import { contexts, strategies, type Evidence } from './answers';
import { EvidenceCard, EvidenceDialog, type SelectedSource } from './components/AnswerSources';
import { useResource } from './resources';

type Result = {
  hits: Evidence[]; evidence: Evidence[]; evidenceTokens: number; retrievalMs: number;
  contextMode: string; trace: Record<string, unknown>[];
};
type Comparison = {label: string; strategy: string; rerank: boolean; result: Result};
const stageLabels: Record<string, string> = {
  lexical_independent: 'BM25 召回', dense_independent: '向量召回', independent_rank: '排名融合',
  initial: '首次召回', relaxed: '放宽匹配', dense_fallback: '向量回退', rerank: '精排',
};

function StrategyFields({name, title, rerank}: {name: string; title: string; rerank: boolean}) {
  const [strategy, setStrategy] = useState(name === 'baseline' ? 'standard' : 'rrf');
  return <fieldset className="strategy-fields">
    <legend>{title}</legend>
    <label>召回<select name={name + '-strategy'} value={strategy} onChange={event => setStrategy(event.target.value)}>
      {Object.entries(strategies).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
    </select></label>
    <label>上下文<select name={name + '-context'} defaultValue={name === 'baseline' ? 'leaf' : 'parent'}>
      {Object.entries(contexts).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
    </select></label>
    {rerank && strategy === 'standard' && <label className="checkbox">
      <input type="checkbox" name={name + '-rerank'} />精排
    </label>}
  </fieldset>;
}

export default function Retrieve({api, rerank}: {api: ApiClient; rerank: boolean}) {
  const datasets = useResource<Dataset[]>(api, '/api/datasets');
  const [results, setResults] = useState<Comparison[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [source, setSource] = useState<SelectedSource>();

  async function compare(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const datasetId = String(form.get('datasetId') ?? '');
    const question = String(form.get('question') ?? '').trim();
    const topK = Number(form.get('topK')), candidateK = Number(form.get('candidateK'));
    if (!datasetId || !question) return;
    if (topK > candidateK) return setError('候选池须不小于最终 Top K。');
    setError(''); setResults([]); setBusy(true);
    const request = {datasetId, question, topK, candidateK, evidenceBudget: Number(form.get('budget'))};
    try {
      for (const [name, label] of [['baseline', '基线'], ['comparison', '对照']]) {
        const strategy = String(form.get(name + '-strategy'));
        const useReranker = form.has(name + '-rerank');
        const result = await api.json<Result>('/api/retrieve/context', 'POST', {
          ...request, strategy, contextMode: form.get(name + '-context'), useReranker,
        });
        setResults(old => [...old, {label, strategy, rerank: useReranker, result}]);
      }
    } catch (error) { setError(messageOf(error)); }
    finally { setBusy(false); }
  }

  return <section className="page">
    <header className="page-heading"><div><h1>检索实验</h1>
      <p className="muted">比较同一知识库、问题和候选预算下的命中与证据。</p></div></header>
    <form className="panel stack" onSubmit={compare}>
      <div className="form-grid">
        <label>知识库<select name="datasetId" required>
          {!datasets.data?.length && <option value="">请选择知识库</option>}
          {datasets.data?.map(item => <option value={item.datasetId} key={item.datasetId}>{item.name}</option>)}
        </select></label>
        <label>检索问题<input name="question" placeholder="输入要对照的问题" maxLength={2000} required /></label>
      </div>
      <div className="form-grid">
        <label>最终 Top K<input name="topK" type="number" defaultValue={8} min={1} max={50} required /></label>
        <label>候选池<input name="candidateK" type="number" defaultValue={64} min={2} max={256} required /></label>
        <label>证据预算（tokens）<input name="budget" type="number" defaultValue={1400} min={1} max={8192} required /></label>
      </div>
      <div className="comparison-grid">
        <StrategyFields name="baseline" title="基线" rerank={rerank} />
        <StrategyFields name="comparison" title="对照" rerank={rerank} />
      </div>
      {(error || datasets.error) && <p className="error" role="alert">{error || datasets.error}</p>}
      <div className="actions"><button className="primary" disabled={busy || !datasets.data?.length}>
        {busy ? '正在运行…' : '运行对照'}</button></div>
    </form>
    <div className="comparison-grid comparison-results" aria-live="polite">
      {results.map(({label, strategy, rerank, result}) => <article key={label} className="panel">
        <h2>{label} · {strategies[strategy]}{rerank && ' + 精排'}</h2>
        <p className="muted">{contexts[result.contextMode]} · {result.hits.length} 个命中 · {result.evidenceTokens} tokens ·
          {' '}{Math.round(result.retrievalMs)} ms</p>
        <h3>召回命中</h3>
        {result.hits.length === 0 && <p className="muted">未召回文档块。</p>}
        {result.hits.map((chunk, index) => <EvidenceCard key={chunk.id ?? index} chunk={chunk} index={index}
          onOriginal={chunk.source && chunk.document_external_id ? () => setSource({chunk}) : undefined} />)}
        <h3>预算内证据</h3>
        {result.evidence.length === 0 && <p className="muted">没有可用证据。</p>}
        {result.evidence.map((chunk, index) => <EvidenceCard key={chunk.id ?? index} chunk={chunk}
          onOriginal={chunk.source && chunk.document_external_id ? () => setSource({chunk}) : undefined} />)}
        {!!result.trace.length && <details className="retrieval-stages">
          <summary>查看检索阶段</summary>
          <ol>{result.trace.map((step, index) => <li key={index}>
            {stageLabels[String(step.stage)] ?? String(step.stage ?? '检索')}
            {typeof step.durationMs === 'number' ? ' · ' + Math.round(step.durationMs) + ' ms' : ''}
          </li>)}</ol>
        </details>}
      </article>)}
    </div>
    {source && <EvidenceDialog api={api} selected={source} onClose={() => setSource(undefined)} />}
  </section>;
}
