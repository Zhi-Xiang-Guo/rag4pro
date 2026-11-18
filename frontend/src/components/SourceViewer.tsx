import { lazy, Suspense, useState } from 'react';
import { sourceRanges } from '../citationRanges';
import type { CitationSpan } from '../answers';
import { HighlightedText } from './HighlightedText';
import { Download } from 'lucide-react';
import { messageOf, type ApiClient } from '../api';
import { sourceUrl, useResource, type SourcePointer, type SourceUnit, type SourceVersion } from '../resources';

const PdfPreview = lazy(() => import('./PdfPreview'));
const DocxPreview = lazy(() => import('./DocxPreview'));

export function unitLabel(kind: string, number: number) {
  switch (kind) {
    case 'page': return '第 ' + number + ' 页';
    case 'paragraph': return '第 ' + number + ' 段';
    case 'table': return '表格 ' + number;
    default: return '正文';
  }
}

export function SourceViewer({api, pointer}: {api: ApiClient; pointer: SourcePointer & {spans?: CitationSpan[]}}) {
  const {data, error, loading} = useResource<SourceVersion>(api, sourceUrl(pointer));
  const [unitKey, setUnitKey] = useState('');
  const [downloadError, setDownloadError] = useState('');
  const [preview, setPreview] = useState(true);
  if (loading) return <p className="muted">正在读取原文…</p>;
  if (error || !data) return <p className="error" role="alert">{error || '原文不可用'}</p>;
  const keyOf = (unit: SourceUnit) => unit.kind + '/' + unit.number;
  const initialKey = pointer.location?.kind + '/' + pointer.location?.unit;
  const unit = data.units.find(item => keyOf(item) === (unitKey || initialKey)) ?? data.units[0];
  const highlights = unit && (!unitKey || unitKey === initialKey) && (!pointer.location || keyOf(unit) === initialKey)
    ? sourceRanges(unit.text, pointer.content, pointer.location?.start, pointer.spans ?? (pointer.span ? [pointer.span] : [])) : [];
  const path = sourceUrl({...pointer, revision: data.revision}, true);
  const visual = data.format === 'pdf' || data.format === 'docx';

  async function download() {
    setDownloadError('');
    try {
      const response = await api.request(path);
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a');
      link.href = url;
      link.download = data!.name;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 15_000);
    } catch (error) { setDownloadError(messageOf(error)); }
  }

  return <div className="source-viewer">
    <div className="actions source-actions"><strong>{data.name}</strong><span className="muted">版本 {data.revision.slice(0, 8)}</span>
      <button onClick={() => void download()}><Download size={14} /> 下载原件</button>
      {visual && <button onClick={() => setPreview(value => !value)}>{preview ? '收起原件' : '查看原件'}</button>}
    </div>
    {downloadError && <p className="error" role="alert">{downloadError}</p>}
    {data.units.length > 1 && unit && <label className="unit-picker">原文位置
      <select value={keyOf(unit)} onChange={e => setUnitKey(e.target.value)}>
        {data.units.map(item => <option key={keyOf(item)} value={keyOf(item)}>{unitLabel(item.kind, item.number)}</option>)}
      </select>
    </label>}
    {unit && <div className="source-text">
      <p className="muted">{unitLabel(unit.kind, unit.number)}{visual ? ' · 提取的正文' : ''}</p>
      <pre tabIndex={0}><HighlightedText text={unit.text} ranges={highlights} /></pre>
    </div>}
    {visual && preview && <Suspense fallback={<p className="muted">正在加载预览…</p>}>
      {data.format === 'pdf' ? <PdfPreview api={api} path={path} page={unit?.number ?? 1} />
        : <DocxPreview api={api} path={path} unit={unit} />}
    </Suspense>}
  </div>;
}
