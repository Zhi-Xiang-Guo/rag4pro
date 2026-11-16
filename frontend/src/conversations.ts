import type {Reference} from './answers';

export type Message = {id: string; role: 'user' | 'assistant'; content: string; model?: string; reference?: Reference};
export type Session = {id: string; title: string; datasetId: string; docId?: string; version?: string; messages: Message[]};
export type History = {sessions: Session[]; currentId: string};

export const historyKey = 'rag4pro.conversations.v1';

function record(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function usableReference(value: unknown): Reference | undefined {
  if (!record(value) || !Array.isArray(value.chunks) || !value.chunks.every(record)) return;
  const chunks = value.chunks.map(chunk => {
    const source = chunk.source;
    return {
      ...chunk,
      content: optionalString(chunk.content),
      content_with_weight: optionalString(chunk.content_with_weight),
      document_name: optionalString(chunk.document_name),
      docnm_kwd: optionalString(chunk.docnm_kwd),
      document_external_id: optionalString(chunk.document_external_id),
      dataset_id: optionalString(chunk.dataset_id),
      section_kwd: optionalString(chunk.section_kwd),
      source: record(source) && typeof source.revision === 'string' && typeof source.kind === 'string'
        && typeof source.unit === 'number' ? {
          revision: source.revision, kind: source.kind, unit: source.unit,
          start: typeof source.start === 'number' ? source.start : undefined,
        } : undefined,
    };
  });
  const spans = Array.isArray(value.spans) ? value.spans.filter(span => {
    return record(span) && Number.isSafeInteger(span.citationIndex) && Number.isSafeInteger(span.start)
      && Number.isSafeInteger(span.end) && typeof span.text === 'string';
  }) : undefined;
  return {chunks, spans};
}

function readMessage(value: unknown): Message | undefined {
  if (!record(value) || typeof value.id !== 'string' || typeof value.content !== 'string'
    || (value.role !== 'assistant' && value.role !== 'user')) return;
  return {
    id: value.id, role: value.role, content: value.content,
    model: optionalString(value.model), reference: usableReference(value.reference),
  };
}

function readSession(value: unknown): Session | undefined {
  if (!record(value) || typeof value.id !== 'string' || typeof value.datasetId !== 'string'
    || !Array.isArray(value.messages)) return;
  return {
    id: value.id, datasetId: value.datasetId, title: optionalString(value.title) || '对话',
    docId: optionalString(value.docId), version: optionalString(value.version),
    messages: value.messages.flatMap(message => readMessage(message) ?? []),
  };
}

export function decodeHistory(stored: string | null): History {
  try {
    const data: unknown = JSON.parse(stored ?? 'null');
    if (record(data) && Array.isArray(data.sessions)) {
      const sessions = data.sessions.flatMap(value => readSession(value) ?? []);
      const currentId = sessions.some(session => session.id === data.currentId) ? String(data.currentId) : '';
      return {sessions, currentId};
    }
  } catch { /* A damaged cache can be discarded. */ }
  return {sessions: [], currentId: ''};
}
