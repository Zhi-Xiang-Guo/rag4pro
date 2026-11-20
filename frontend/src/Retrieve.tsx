import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { messageOf, withQuery, type ApiClient, type Dataset } from './api';
import { contexts, excerpt, strategies } from './answers';
import { EvidenceCard, EvidenceDialog, type SelectedSource } from './components/AnswerSources';
import { useResource, type DocumentRow } from './resources';
import { buildExperimentReport, compareChunks, describeStage, type RetrievalResult } from './retrievalComparison';
import { downloadExperiment, readRecipe, type RetrievalArm, type RetrievalRecipe } from './retrievalRecipe';

type Comparison = {label: string; strategy: string; rerank: boolean} & (
  {result: RetrievalResult; error?: never} | {result?: never; error: string}
);
const arms = [['baseline', '基线'], ['comparison', '对照']] as const;

function StrategyFields({name, title, rerank, initial}: {
  name: string; title: string; rerank: boolean; initial?: RetrievalArm;
}) {
  const [strategy, setStrategy] = useState<string>(initial?.strategy ?? (name === 'baseline' ? 'standard' : 'rrf'));
  return <fieldset className="strategy-fields">
    <legend>{title}</legend>
    <label>召回<select name={name + '-strategy'} value={strategy} onChange={event => setStrategy(event.target.value)}>
      {Object.entries(strategies).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
    </select></label>
    <label>上下文<select name={name + '-context'} defaultValue={initial?.contextMode ?? 'leaf'}>
      {Object.entries(contexts).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
    </select></label>
    {(rerank || initial?.useReranker) && strategy === 'standard' && <label className="checkbox">
      <input type="checkbox" name={name + '-rerank'} defaultChecked={initial?.useReranker} />精排
    </label>}
  </fieldset>;
}

function ComparisonSummary({baseline, comparison}: {baseline: RetrievalResult; comparison: RetrievalResult}) {
  const hits = compareChunks(baseline.hits, comparison.hits);
  const evidence = compareChunks(baseline.evidence, comparison.evidence);
  return <section className="panel stack">
    <h2>命中与证据差异</h2>
    <div className="ui-table-wrap"><table className="ui-table">
      <thead><tr><th scope="col">范围</th><th scope="col">共同</th><th scope="col">基线独有</th>
        <th scope="col">对照独有</th></tr></thead>
      <tbody>
        <tr><th scope="row">Top K 命中</th><td>{hits.shared}</td><td>{hits.baselineOnly}</td>
          <td>{hits.comparisonOnly}</td></tr>
        <tr><th scope="row">预算内证据</th><td>{evidence.shared}</td><td>{evidence.baselineOnly}</td>
          <td>{evidence.comparisonOnly}</td></tr>
      </tbody>
    </table></div>
    {hits.changes.length > 0 && <details className="retrieval-stages">
      <summary>查看命中块名次</summary>
      <div className="ui-table-wrap"><table className="ui-table">
        <thead><tr><th scope="col">文档块</th><th scope="col">基线名次</th><th scope="col">对照名次</th></tr></thead>
        <tbody>{hits.changes.map(({id, chunk, baseline, comparison}) => {
          const content = excerpt(chunk).replace(/\s+/g, ' ').trim();
          return <tr key={id}>
            <td>{chunk.document_name ?? chunk.docnm_kwd ?? id}
              {content && <div className="muted">{content.length > 96 ? content.slice(0, 96) + '…' : content}</div>}
            </td>
            <td>{baseline ?? '—'}</td><td>{comparison ?? '—'}</td>
          </tr>;
        })}</tbody>
      </table></div>
    </details>}
  </section>;
}

export default function Retrieve({api, rerank}: {api: ApiClient; rerank: boolean}) {
  const [params, setParams] = useSearchParams();
  const datasets = useResource<Dataset[]>(api, '/api/datasets');
  const requestedDataset = params.get('datasetId');
  const datasetId = (requestedDataset ?? datasets.data?.[0]?.datasetId) || '';
  const datasetAvailable = !!datasets.data?.some(dataset => dataset.datasetId === datasetId);
  const docId = params.get('docId') ?? '';
  const documents = useResource<DocumentRow[]>(api, datasetAvailable
    ? withQuery('/api/documents', {datasetId}) : undefined);
  const documentAvailable = !docId || !!documents.data?.some(document => document.docId === docId);
  const [lastRun, setLastRun] = useState<{recipe: RetrievalRecipe; startedAt: string; scopeName: string}>();

  const [results, setResults] = useState<Comparison[]>([]);
  const baselineResult = results[0]?.result;
  const comparisonResult = results[1]?.result;
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [source, setSource] = useState<SelectedSource>();
  const [imported, setImported] = useState<{revision: number; recipe: RetrievalRecipe}>();
  const formRef = useRef<HTMLFormElement>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const activeRun = useRef<AbortController>();

  useEffect(() => () => activeRun.current?.abort(), []);

  function cancel() {
    activeRun.current?.abort();
    activeRun.current = undefined;
    setBusy(false);
  }

  function currentRecipe(formElement: HTMLFormElement) {
    if (!datasetAvailable || !documentAvailable) throw new Error('请先选择可用的知识库和文档范围。');
    const form = new FormData(formElement);
    const armFields = (name: string) => ({
      strategy: form.get(name + '-strategy'),
      contextMode: form.get(name + '-context'),
      useReranker: form.has(name + '-rerank'),
    });
    return readRecipe({
      version: 1, datasetId, docId: docId || undefined, question: form.get('question'),
      topK: Number(form.get('topK')), candidateK: Number(form.get('candidateK')),
      evidenceBudget: Number(form.get('budget')),
      baseline: armFields('baseline'), comparison: armFields('comparison'),
    });
  }

  function exportRecipe() {
    if (!formRef.current?.reportValidity()) return;
    try {
      downloadExperiment('rag4pro-retrieval-recipe.json', currentRecipe(formRef.current));
      setError('');
    } catch (error) { setError(messageOf(error)); }
  }

  async function importRecipe(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    try {
      if (file.size > 16384) throw new Error('配置文件不能超过 16 KB。');
      const recipe = readRecipe(JSON.parse(await file.text()));
      if (activeRun.current) return;
      setImported(old => ({revision: (old?.revision ?? 0) + 1, recipe}));
      setParams(recipe.docId ? {datasetId: recipe.datasetId, docId: recipe.docId} : {datasetId: recipe.datasetId});
      setError('');
    } catch (error) { setError('导入失败。' + messageOf(error)); }
  }

  async function compare(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (activeRun.current) return;
    let recipe: RetrievalRecipe;
    try {
      recipe = currentRecipe(event.currentTarget);
      if (!rerank && (recipe.baseline.useReranker || recipe.comparison.useReranker)) {
        throw new Error('当前服务未启用精排，请取消勾选精排或配置精排模型。');
      }
    } catch (error) {
      setError(messageOf(error));
      return;
    }
    const controller = new AbortController();
    activeRun.current = controller;
    setError('');
    setSource(undefined);
    setResults([]);
    setBusy(true);
    const request = {
      datasetId: recipe.datasetId, docId: recipe.docId, question: recipe.question,
      topK: recipe.topK, candidateK: recipe.candidateK, evidenceBudget: recipe.evidenceBudget,
    };
    const datasetName = datasets.data?.find(dataset => dataset.datasetId === request.datasetId)?.name;
    const documentTitle = documents.data?.find(document => document.docId === request.docId)?.title;
    setLastRun({recipe, startedAt: new Date().toISOString(),
      scopeName: [datasetName ?? request.datasetId, documentTitle ?? request.docId].filter(Boolean).join(' / ')});
    try {
      for (const [name, label] of arms) {
        const arm = recipe[name];
        const {strategy, useReranker} = arm;
        const row = {label, strategy, rerank: useReranker};
        let outcome: Comparison;
        try {
          const result = await api.json<RetrievalResult>('/api/retrieve/context', 'POST', {
            ...request, ...arm,
          }, controller.signal);
          outcome = {...row, result};
        } catch (error) {
          outcome = {...row, error: messageOf(error)};
        }
        if (controller.signal.aborted) return;
        setResults(old => [...old, outcome]);
      }
    } finally {
      if (activeRun.current === controller) {
        activeRun.current = undefined;
        setBusy(false);
      }
    }
  }

  return <section className="page">
    <header className="page-heading"><div><h1>检索实验</h1>
      <p className="muted">比较同一知识库、问题和候选预算下的命中与证据。</p></div></header>
    <form className="panel stack" key={imported?.revision} ref={formRef} onSubmit={compare}>
      <div className="form-grid">
        <label>知识库<select value={datasetId} onChange={event => setParams({datasetId: event.target.value})} required>
          {!datasetAvailable && <option value={datasetId}>{datasetId && datasets.data ? '知识库不可用' : '请选择知识库'}</option>}
          {datasets.data?.map(item => <option value={item.datasetId} key={item.datasetId}>{item.name}</option>)}
        </select></label>
        <label>文档<select value={docId} disabled={!datasetAvailable || documents.loading}
          onChange={event => setParams(event.target.value ? {datasetId, docId: event.target.value} : {datasetId})}>
          <option value="">全部文档</option>
          {!documentAvailable && <option value={docId}>{documents.loading ? '正在读取文档…' : '文档不可用'}</option>}
          {documents.data?.map(document => <option value={document.docId} key={document.docId}>{document.title}</option>)}
        </select></label>
        <label>检索问题<input name="question" defaultValue={imported?.recipe.question} placeholder="输入要对照的问题" maxLength={2000} required /></label>
      </div>
      <div className="form-grid">
        <label>最终 Top K<input name="topK" type="number" defaultValue={imported?.recipe.topK ?? 8} min={1} max={50} required /></label>
        <label>候选池<input name="candidateK" type="number" defaultValue={imported?.recipe.candidateK ?? 64} min={2} max={256} required /></label>
        <label>证据预算（tokens）<input name="budget" type="number" defaultValue={imported?.recipe.evidenceBudget ?? 1400} min={1} max={8192} required /></label>
      </div>
      <div className="comparison-grid">
        {arms.map(([name, title]) =>
          <StrategyFields key={name} name={name} title={title} rerank={rerank} initial={imported?.recipe[name]} />)}
      </div>
      {(error || datasets.error || documents.error) && <p className="error" role="alert">
        {error || datasets.error || documents.error}</p>}
      <div className="actions"><button className="primary" disabled={busy || !datasetAvailable || !documentAvailable}>
        {busy ? '正在运行…' : '运行对照'}</button>
        {busy && <button type="button" onClick={cancel}>停止等待</button>}
        <button type="button" disabled={busy} onClick={() => importInput.current?.click()}>导入配置</button>
        <button type="button" disabled={busy || !datasetAvailable || !documentAvailable} onClick={exportRecipe}
          title="配置文件包含检索问题和范围">导出配置</button>
        <input type="file" hidden accept=".json,application/json" ref={importInput} onChange={importRecipe} />
      </div>
    </form>
    {lastRun && results.length > 0 && <section className="panel stack">
      <h2>本次实验</h2>
      <p>{lastRun.recipe.question}</p>
      <p className="muted">{lastRun.scopeName} · Top {lastRun.recipe.topK} · 候选 {lastRun.recipe.candidateK} ·
        {' '}证据预算 {lastRun.recipe.evidenceBudget} tokens</p>
      <div className="actions">
        <Link className="button" to={withQuery('/chat', {
          datasetId: lastRun.recipe.datasetId, docId: lastRun.recipe.docId,
        })}>按此范围提问</Link>
        {baselineResult && comparisonResult &&
          <button type="button" onClick={() => downloadExperiment('rag4pro-retrieval-report.json',
            buildExperimentReport(lastRun.recipe, lastRun.startedAt, baselineResult, comparisonResult))}>
            导出结果
          </button>}
      </div>
    </section>}
    {baselineResult && comparisonResult && <ComparisonSummary baseline={baselineResult} comparison={comparisonResult} />}
    <div className="comparison-grid comparison-results" aria-live="polite">
      {results.map(({label, strategy, rerank, result, error}) => <article key={label} className="panel">
        <h2>{label} · {strategies[strategy]}{rerank && ' + 精排'}</h2>
        {result ? <>
        <p className="muted">{contexts[result.contextMode]} · {result.hits.length} 个命中 ·
          {' '}{result.evidence.length} 个证据块 · {result.evidenceTokens} tokens · 召回 {Math.round(result.retrievalMs)} ms</p>
        <p className="muted">{strategy === 'standard'
          ? (rerank ? '得分为模型与词法的加权分。' : '得分为候选向量与词法的加权分。')
          : '得分按召回名次计算。'}</p>
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
          <ol>{result.trace.map((step, index) => {
            const {label, details} = describeStage(step);
            return <li key={index}>{[label, ...details].join(' · ')}</li>;
          })}</ol>
        </details>}
        </> : <p className="error">{error}</p>}
      </article>)}
    </div>
    {source && <EvidenceDialog api={api} selected={source} onClose={() => setSource(undefined)} />}
  </section>;
}
