/* Knowledge navigation and table interactions adapted from the Ragent frontend (Apache-2.0). */
import { useState, type FormEvent } from 'react';
import { Link, Route, Routes, useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, BookOpen, ChevronRight, FilePlus2, Plus, Search } from 'lucide-react';
import { messageOf, withQuery, type ApiClient, type Dataset } from './api';
import { Modal } from './components/Modal';
import { Table } from './components/Table';
import { SourceViewer, unitLabel } from './components/SourceViewer';
import {
  chunkStrategies, documentUrl, isPending, jobLabels, knowledgeUrl, sourceUrl, useDocuments, useResource,
  type DocumentRow, type IndexChunk, type Job, type SourcePointer, type SourceVersion,
} from './resources';

type ChunkOptions = {chunkStrategy?: string; chunkTokenSize?: number; overlapPercent?: number};

function ChunkFields({options}: {options?: ChunkOptions}) {
  const [strategy, setStrategy] = useState(options?.chunkStrategy || 'parent_child');
  return <div className="form-grid">
    <label>分块策略<select name="strategy" value={strategy} onChange={e => setStrategy(e.target.value)}>
      {Object.entries(chunkStrategies).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
    </select></label>
    <label>分块大小（tokens）<input type="number" min={32} max={1024} name="size" required defaultValue={options?.chunkTokenSize ?? 160} /></label>
    {strategy === 'fixed' && <label>窗口重叠（%）<input type="number" min={0} max={50} name="overlap"
      required defaultValue={options?.overlapPercent ?? 0} /></label>}
  </div>;
}

function chunkOptions(form: FormData) {
  return {
    chunkStrategy: String(form.get('strategy') || 'parent_child'),
    chunkTokenSize: Number(form.get('size') ?? 160),
    overlapPercent: Number(form.get('overlap') || 0),
  };
}

function DatasetEditor({api, dataset, close, done}: {
  api: ApiClient; dataset?: Dataset; close: () => void; done: (datasetId: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const datasetId = dataset?.datasetId || String(form.get('id')).trim();
    setBusy(true);
    setError('');
    try {
      await api.json('/api/datasets', 'POST', {
        datasetId, name: String(form.get('name')).trim(), language: form.get('language'), ...chunkOptions(form),
      });
      done(datasetId);
    } catch (error) { setError(messageOf(error)); }
    finally { setBusy(false); }
  }

  return <Modal title={dataset ? '知识库设置' : '新建知识库'} onClose={close}>
    <form onSubmit={submit} className="stack">
      <label>名称<input name="name" defaultValue={dataset?.name} placeholder="例如 客户支持手册" maxLength={100} required /></label>
      <label>知识库 ID<input name="id" defaultValue={dataset?.datasetId} readOnly={!!dataset}
        pattern="[a-z0-9][a-z0-9_\-]{0,63}" placeholder="例如 support-guides" required /></label>
      <label>文档语言<select name="language" defaultValue={dataset?.language ?? ''}>
        <option value="">自动</option><option value="Chinese">中文</option><option value="English">英文</option>
      </select></label>
      <ChunkFields options={dataset} />
      {error && <p className="error" role="alert">{error}</p>}
      <div className="actions modal-actions"><button type="button" onClick={close}>取消</button>
        <button className="primary" disabled={busy}>{busy ? '保存中…' : '保存'}</button></div>
    </form>
  </Modal>;
}

function DocumentEditor({api, dataset, document, source, close, done}: {
  api: ApiClient; dataset?: Dataset; document?: DocumentRow; source?: SourceVersion;
  close: () => void; done: () => void;
}) {
  const [file, setFile] = useState<File>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [preview, setPreview] = useState<{total: number; parentCount: number; sectionCount: number;
    pieces: {text: string; tokens: number}[]}>();
  const canPaste = !source || ['txt', 'md'].includes(source.format);

  async function previewText(event: React.MouseEvent<HTMLButtonElement>) {
    const form = new FormData(event.currentTarget.form!);
    setError('');
    setBusy(true);
    try {
      if (file && !/\.(txt|md)$/i.test(file.name)) throw new Error('分块预览可用于 TXT 和 Markdown。');
      const text = file ? await file.text() : String(form.get('text') ?? '');
      if (!text.trim()) throw new Error('请先输入文档正文。');
      const format = file ? (file.name.toLowerCase().endsWith('.md') ? 'md' : 'txt') : String(form.get('format'));
      setPreview(await api.json('/api/chunk-preview', 'POST', {
        datasetId: dataset?.datasetId ?? document?.datasetId,
        title: String(form.get('title') || file?.name || '文档'), text, fileType: format, ...chunkOptions(form),
      }));
    } catch (error) { setError(messageOf(error)); }
    finally { setBusy(false); }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const datasetId = dataset?.datasetId ?? document?.datasetId;
    if (!datasetId) return;
    setError('');
    setBusy(true);
    try {
      const pasted = String(form.get('text') ?? '');
      const title = String(form.get('title') ?? '').trim();
      let name = file?.name;
      let bytes: Blob | undefined = file;
      if (!bytes && pasted.trim()) {
        const format = String(form.get('format') || 'txt');
        name = (title || '文档') + '.' + format;
        bytes = new Blob([pasted], {type: 'text/plain;charset=utf-8'});
      }
      if (!bytes && source && document && !canPaste) {
        name = source.name;
        const original = await api.request(sourceUrl({datasetId, docId: document.docId, revision: source.revision}, true));
        bytes = await original.blob();
      }
      if (!bytes || !name) throw new Error('请选择文件或输入文本。');
      if (bytes.size > 16 * 1024 * 1024) throw new Error('文件超过 16 MiB。');
      const options = chunkOptions(form);
      await api.request(withQuery('/api/ingestion', {
        datasetId, docId: document?.docId, filename: name, title: title || name,
        chunkStrategy: options.chunkStrategy, chunkTokenSize: String(options.chunkTokenSize),
        overlapPercent: String(options.overlapPercent),
      }), {method: 'POST', body: bytes, headers: {'Content-Type': 'application/octet-stream'}});
      done();
    } catch (error) { setError(messageOf(error)); }
    finally { setBusy(false); }
  }

  return <Modal title={document ? '修订文档与重建索引' : '导入文档'} onClose={close}>
    <form className="stack" onSubmit={submit}>
      <label>标题<input name="title" defaultValue={document?.title} maxLength={255} placeholder="留空使用文件名" /></label>
      <label>文件<input type="file" accept=".txt,.md,.pdf,.docx" onChange={e => {setFile(e.target.files?.[0]); setPreview(undefined);}} /></label>
      <span className="muted">支持 UTF-8 TXT、Markdown、带文字的 PDF 和 DOCX，最多 16 MiB。</span>
      {canPaste && !file && <>
        <label>文本格式<select name="format" defaultValue={source?.format === 'md' ? 'md' : 'txt'}>
          <option value="txt">TXT</option><option value="md">Markdown</option></select></label>
        <label>正文<textarea name="text" rows={6} defaultValue={source?.units[0]?.text} />
        </label>
      </>}
      <ChunkFields options={document ?? dataset} />
      {error && <p className="error" role="alert">{error}</p>}
      {preview && <div className="chunk-preview">
        <strong>{preview.total} 个检索块 · {preview.parentCount} 个父段 · {preview.sectionCount} 个章节</strong>
        {preview.pieces.slice(0, 8).map((piece, i) => <p key={i}>{piece.text.slice(0, 220)}</p>)}
      </div>}
      <div className="actions modal-actions">
        <button type="button" onClick={close}>取消</button>
        {canPaste && <button type="button" disabled={busy} onClick={event => void previewText(event)}>预览分块</button>}
        <button className="primary" disabled={busy}>{document ? '提交重建' : '上传并建立索引'}</button>
      </div>
    </form>
  </Modal>;
}

function StatusChip({job, document}: {job?: Job; document?: DocumentRow}) {
  const status = job?.status ?? (document ? 'READY' : 'QUEUED');
  return <span className={'status-chip ' + status.toLowerCase()}>{jobLabels[status]}</span>;
}

function KnowledgeList({api}: {api: ApiClient}) {
  const navigate = useNavigate();
  const [filter, setFilter] = useState('');
  const [create, setCreate] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const datasets = useResource<Dataset[]>(api, '/api/datasets', refresh);
  const search = filter.trim().toLowerCase();
  const rows = datasets.data?.filter(item => (item.name + ' ' + item.datasetId).toLowerCase().includes(search)) ?? [];
  async function demo() {
    setBusy(true);
    setError('');
    try {
      await api.json('/api/demo', 'POST', {});
      setRefresh(value => value + 1);
    } catch (error) { setError(messageOf(error)); }
    finally { setBusy(false); }
  }

  return <section className="page">
    <header className="page-heading">
      <h1>知识库</h1><button className="primary" onClick={() => setCreate(true)}><Plus size={16} /> 新建知识库</button>
    </header>
    {(datasets.error || error) && <p className="error" role="alert">{datasets.error || error}</p>}
    <div className="filters actions">
      <label className="search-field"><Search size={16} /><input aria-label="搜索知识库" placeholder="搜索名称或 ID" value={filter}
        onChange={event => setFilter(event.target.value)} /></label>
      <button disabled={busy} onClick={() => void demo()}>载入支付手册示例</button>
    </div>
    <Table>
      <thead><tr><th>知识库</th><th>文档数</th><th>分块策略</th><th>操作</th></tr></thead>
      <tbody>{rows.map(item => <tr key={item.datasetId}>
        <td><Link to={knowledgeUrl(item.datasetId)} className="strong">{item.name}</Link><div className="muted">{item.datasetId}</div></td>
        <td>{item.documentCount}</td><td>{chunkStrategies[item.chunkStrategy ?? 'general']}</td>
        <td><Link to={knowledgeUrl(item.datasetId)}>管理文档 <ChevronRight size={13} /></Link></td>
      </tr>)}</tbody>
    </Table>
    {datasets.loading && <p className="empty">正在加载…</p>}
    {!datasets.loading && !rows.length && !datasets.error && <p className="empty"><BookOpen size={18} /> 暂无匹配的知识库</p>}
    {create && <DatasetEditor api={api} close={() => setCreate(false)} done={id => navigate(knowledgeUrl(id))} />}
  </section>;
}

function Library({api}: {api: ApiClient}) {
  const {datasetId = ''} = useParams();
  const navigate = useNavigate();
  const [refresh, setRefresh] = useState(0);
  const [mode, setMode] = useState<'upload' | 'settings'>();
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('');
  const dataset = useResource<Dataset[]>(api, '/api/datasets', refresh).data?.find(item => item.datasetId === datasetId);
  const records = useDocuments(api, datasetId, refresh);
  const latestJobs = new Map<string, Job>();
  for (const job of records.jobs) if (!latestJobs.has(job.docId)) latestJobs.set(job.docId, job);
  const ids = new Set([...records.documents.map(row => row.docId),
    ...[...latestJobs.values()].filter(job => job.status !== 'READY' && !job.expectedVersion).map(job => job.docId)]);
  const rows = [...ids].map(id => ({
    id, document: records.documents.find(row => row.docId === id), job: latestJobs.get(id),
  })).filter(row => {
    const title = row.document?.title || row.job?.title || row.id;
    return title.toLowerCase().includes(filter.trim().toLowerCase());
  });

  function done() { setMode(undefined); setRefresh(value => value + 1); }

  async function deleteDataset() {
    if (!window.confirm('删除这个空知识库？')) return;
    try {
      await api.json(withQuery('/api/datasets', {datasetId}), 'DELETE');
      navigate('/knowledge');
    } catch (error) { setError(messageOf(error)); }
  }

  return <section className="page">
    <nav className="breadcrumbs"><Link to="/knowledge"><ArrowLeft size={14} /> 知识库</Link><ChevronRight size={13} /><span>{dataset?.name ?? datasetId}</span></nav>
    <header className="page-heading">
      <div><h1>{dataset?.name ?? datasetId}</h1><p className="muted">文档与索引</p></div>
      <div className="actions">
        <Link to={withQuery('/chat', {datasetId})} className="button">去提问</Link>
        {dataset?.registered && <button onClick={() => setMode('settings')}>分块设置</button>}
        {dataset?.registered && !records.documents.length && !records.jobs.some(isPending)
          && <button onClick={() => void deleteDataset()}>删除空库</button>}
        <button className="primary" disabled={!dataset} onClick={() => setMode('upload')}><FilePlus2 size={17} /> 导入文档</button>
      </div>
    </header>
    {(error || records.error) && <p className="error" role="alert">{error || records.error}</p>}
    <div className="filters actions"><label className="search-field"><Search size={16} />
      <input aria-label="搜索文档" placeholder="搜索文档" value={filter} onChange={event => setFilter(event.target.value)} /></label>
      <button onClick={() => setRefresh(value => value + 1)}>刷新</button></div>
    <Table>
      <thead><tr><th>文档</th><th>索引状态</th><th>检索块</th><th>分块策略</th><th>操作</th></tr></thead>
      <tbody>{rows.map(({id, document, job}) => <tr key={id}>
        <td><Link className="strong" to={documentUrl(datasetId, id)}>{document?.title || job?.title || id}</Link>
          <div className="muted">{document?.sourceName ?? job?.filename}</div></td>
        <td><StatusChip document={document} job={job} /></td><td>{document?.chunkCount ?? '—'}</td>
        <td>{document ? chunkStrategies[document.chunkStrategy ?? 'general'] : '—'}</td>
        <td><Link to={documentUrl(datasetId, id)}>查看详情 <ChevronRight size={13} /></Link></td>
      </tr>)}</tbody>
    </Table>
    {!records.error && !rows.length && <p className="empty">{records.loading ? '正在读取文档…' : '暂无匹配的文档'}</p>}
    {mode === 'settings' && dataset && <DatasetEditor api={api} dataset={dataset} close={() => setMode(undefined)} done={done} />}
    {mode === 'upload' && dataset && <DocumentEditor api={api} dataset={dataset} close={() => setMode(undefined)} done={done} />}
  </section>;
}

function DocumentDetail({api}: {api: ApiClient}) {
  const {datasetId = '', docId = ''} = useParams();
  const navigate = useNavigate();
  const [refresh, setRefresh] = useState(0);
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState('');
  const [tab, setTab] = useState<'chunks' | 'source'>('chunks');
  const [selected, setSelected] = useState('');
  const records = useDocuments(api, datasetId, refresh);
  const document = records.documents.find(row => row.docId === docId);
  const job = records.jobs.find(row => row.docId === docId);
  const latest = document?.sourceRevision;
  const pointer: SourcePointer = {datasetId, docId, revision: latest};
  const source = useResource<SourceVersion>(api, editing && latest ? sourceUrl(pointer) : undefined);
  const chunks = useResource<IndexChunk[]>(api, document ? withQuery('/api/chunks', {
    datasetId, docId, context: 'true', revision: latest,
  }) : undefined, refresh);
  const rows = (chunks.data ?? []).slice().sort((a, b) =>
    (a.chunk_order_int ?? 0) - (b.chunk_order_int ?? 0));
  const selectedChunk = rows.find(row => row.id === selected);
  const selection = selectedChunk?.source_revision_kwd && selectedChunk.source_kind_kwd
    ? {revision: selectedChunk.source_revision_kwd, kind: selectedChunk.source_kind_kwd,
      unit: selectedChunk.source_unit_int ?? 1, start: selectedChunk.source_start_int}
    : undefined;

  async function retry() {
    setError('');
    try {
      await api.json(withQuery('/api/ingestion/retry', {jobId: job?.id}), 'POST');
      setRefresh(value => value + 1);
    } catch (error) { setError(messageOf(error)); }
  }
  async function remove() {
    if (!window.confirm('移除此文档及其检索索引？')) return;
    try {
      await api.json(withQuery('/api/documents', {datasetId, docId}), 'DELETE');
      navigate(knowledgeUrl(datasetId));
    } catch (error) { setError(messageOf(error)); }
  }

  const editorDocument = document ?? (job ? {
    datasetId, docId, title: job.title, chunkCount: 0,
  } : undefined);
  return <section className="page">
    <nav className="breadcrumbs"><Link to="/knowledge">知识库</Link><ChevronRight size={13} />
      <Link to={knowledgeUrl(datasetId)}>{datasetId}</Link><ChevronRight size={13} /><span>文档详情</span></nav>
    <header className="page-heading">
      <div><h1>{document?.title ?? job?.title ?? docId}</h1>{(document || job) && <StatusChip document={document} job={job} />}</div>
      <div className="actions">
        {document && <Link to={withQuery('/chat', {datasetId, docId})} className="button">就此文档提问</Link>}
        {editorDocument && <button className="primary" disabled={isPending(job)} onClick={() => setEditing(true)}>
          {latest ? '修订与重建' : '上传原件'}</button>}
        {document && <button disabled={isPending(job)} onClick={() => void remove()}>移除文档</button>}
      </div>
    </header>
    {(error || records.error) && <p className="error" role="alert">{error || records.error}</p>}
    {job?.status === 'FAILED' && <div className="actions error" role="alert">
      <span>{job.error}{document && '。当前版本仍可检索。'}</span><button onClick={() => void retry()}>重试入库</button>
    </div>}
    {!document && !records.loading && <p className="empty">{job ? '索引完成后即可查看分块。' : '找不到此文档。'}</p>}
    {document && <><div className="tabs">
      <button className={tab === 'chunks' ? 'active' : ''} onClick={() => setTab('chunks')}>分块</button>
      <button className={tab === 'source' ? 'active' : ''} onClick={() => setTab('source')}>原文</button>
    </div>
    {tab === 'source' ? (
      <div className="panel">{latest
        ? <SourceViewer key={(selection?.revision ?? latest) + ':' + selected} api={api}
          pointer={{...pointer, location: selection, content: selectedChunk?.content_with_weight}} />
        : <p className="muted">这份旧文档有索引块。上传原件后可预览和修订。</p>}</div>
    ) : <>
      {chunks.error && <p className="error" role="alert">{chunks.error}</p>}
      {chunks.loading && <p className="empty">正在读取分块…</p>}
      <Table>
        <thead><tr><th>块类型</th><th>来源</th><th>文本预览</th><th>操作</th></tr></thead>
        <tbody>{rows.map(row => <tr key={row.id}>
          <td>{row.context_level_kwd === 'parent' ? '父段' : row.context_level_kwd === 'section' ? '章节' : '检索块'}</td>
          <td>{row.source_kind_kwd ? unitLabel(row.source_kind_kwd, row.source_unit_int ?? 1) : row.section_kwd || '—'}</td>
          <td className="snippet">{row.content_with_weight?.slice(0, 150)}</td>
          <td><button onClick={() => setSelected(row.id)}>查看块</button></td>
        </tr>)}</tbody>
      </Table>
      {selectedChunk && <div className="panel selected-chunk">
        <div className="actions"><strong>分块正文</strong>
          {latest && <button onClick={() => setTab('source')}>在原文中查看</button>}
          {selectedChunk.mom_id && <button onClick={() => setSelected(selectedChunk.mom_id!)}>查看父段</button>}
          {selectedChunk.section_id && selectedChunk.section_id !== selectedChunk.id
            && <button onClick={() => setSelected(selectedChunk.section_id!)}>查看章节</button>}
        </div>
        <pre tabIndex={0}>{selectedChunk.content_with_weight}</pre>
      </div>}
    </>}
    </>}
    {editing && editorDocument && (source.loading
      ? <Modal title="读取原件" onClose={() => setEditing(false)}><p>正在读取…</p></Modal>
      : <DocumentEditor key={latest ?? docId} api={api} document={editorDocument} source={source.data}
        close={() => setEditing(false)} done={() => { setEditing(false); setRefresh(value => value + 1); }} />)}
  </section>;
}

export default function Knowledge({api}: {api: ApiClient}) {
  return <Routes>
    <Route index element={<KnowledgeList api={api} />} />
    <Route path=":datasetId" element={<Library api={api} />} />
    <Route path=":datasetId/documents/:docId" element={<DocumentDetail api={api} />} />
  </Routes>;
}
