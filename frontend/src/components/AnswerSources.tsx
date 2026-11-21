/* Evidence panel adapted from Ragent's source review flow under Apache-2.0. */
import { useState } from 'react';
import { FileText } from 'lucide-react';
import type { ApiClient } from '../api';
import { excerpt, type CitationSpan, type Evidence, type Reference } from '../answers';
import { Modal } from './Modal';
import { SourceViewer, unitLabel } from './SourceViewer';

export type SelectedSource = {chunk: Evidence; span?: CitationSpan; datasetId?: string};

export function EvidenceCard({chunk, span, index, onOriginal}: {
  chunk: Evidence; span?: CitationSpan; index?: number; onOriginal?: () => void;
}) {
  const body = excerpt(chunk);
  const exact = span && span.start >= 0 && span.end >= span.start
    && body.slice(span.start, span.end) === span.text;
  const title = chunk.document_name ?? chunk.docnm_kwd ?? '文档片段';
  return <article className="evidence-card">
    <header><FileText size={15} /><strong>{index !== undefined && '[' + (index + 1) + '] '}{title}</strong></header>
    {(chunk.source || chunk.section_kwd) && <p className="muted">
      {chunk.source ? unitLabel(chunk.source.kind, chunk.source.unit) : chunk.section_kwd}
      {chunk.context_level_kwd === 'parent' ? ' · 父段' : chunk.context_level_kwd === 'section' ? ' · 章节' : ''}
    </p>}
    {typeof chunk.similarity === 'number' && <small className="muted">得分 {chunk.similarity.toFixed(4)}</small>}
    <pre tabIndex={0}>{exact && span ? <>{body.slice(0, span.start)}<mark>{body.slice(span.start, span.end)}</mark>
      {body.slice(span.end)}</> : body}</pre>
    {onOriginal && <button onClick={onOriginal}>查看原文</button>}
  </article>;
}

export function EvidenceDialog({api, selected, onClose}: {api: ApiClient; selected: SelectedSource; onClose: () => void}) {
  const {chunk, span, datasetId} = selected;
  const source = chunk.source;
  return <Modal title="查阅引用" onClose={onClose} wide>
    <EvidenceCard chunk={chunk} span={span} />
    {source && chunk.document_external_id && (chunk.dataset_id || datasetId)
      && <SourceViewer api={api} pointer={{
        datasetId: chunk.dataset_id ?? datasetId!, docId: chunk.document_external_id,
        revision: source.revision, location: source, content: excerpt(chunk), span,
      }} />}
  </Modal>;
}

export function AnswerSources({api, reference, selectedIndex, datasetId}: {
  api: ApiClient; reference?: Reference; selectedIndex?: number; datasetId?: string;
}) {
  const [reading, setReading] = useState<SelectedSource>();
  const all = reference?.chunks ?? [];
  const indices = selectedIndex === undefined ? all.map((_, index) => index) : [selectedIndex];
  return <section className="answer-sources" aria-label="引用证据">
    <h2>引用证据</h2>
    {!all.length && <p className="muted">有引用的回答会在这里显示来源。</p>}
    {indices.map(index => {
      const chunk = all[index];
      if (!chunk) return null;
      const span = reference?.spans?.find(item => item.citationIndex === index);
      return <EvidenceCard key={index} chunk={chunk} span={span} index={index}
        onOriginal={chunk.source && chunk.document_external_id ? () => setReading({chunk, span, datasetId}) : undefined} />;
    })}
    {reading && <EvidenceDialog api={api} selected={reading} onClose={() => setReading(undefined)} />}
  </section>;
}
