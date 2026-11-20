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

export function recentHistory(messages: readonly Message[], question: string) {
  const completed: Pick<Message, 'role' | 'content'>[][] = [];
  let user: Message | undefined;
  for (const message of messages) {
    if (message.role === 'user') {
      user = message;
    } else if (user) {
      completed.push([
        {role: 'user', content: user.content},
        {role: 'assistant', content: message.content},
      ]);
      user = undefined;
    }
  }
  return [...completed.slice(-5).flat(), {role: 'user' as const, content: question}];
}

export function encodeHistory(history: History, budget = 2_000_000): string {
  const prioritized = [...history.sessions.filter(session => session.id === history.currentId),
    ...history.sessions.filter(session => session.id !== history.currentId)];
  const currentId = prioritized[0]?.id === history.currentId ? history.currentId : '';
  const prefix = '{"currentId":' + JSON.stringify(currentId) + ',"sessions":[';
  const saved: string[] = [];
  let remaining = budget - prefix.length - 2;

  for (const session of prioritized) {
    let entry = JSON.stringify(session);
    if (entry.length > remaining && !saved.length) {
      // A single long conversation can exceed the quota because it holds snapshots of every cited chunk.
      const messages = session.messages.map(({reference: _reference, ...message}) => message);
      entry = JSON.stringify({...session, messages});
      if (entry.length > remaining) {
        entry = JSON.stringify({...session, messages: messages.slice(-12)});
      }
    }
    if (entry.length > remaining) continue;
    saved.push(entry);
    remaining -= entry.length + 1;
  }
  return saved.length ? prefix + saved.join(',') + ']}'
    : JSON.stringify({currentId: '', sessions: []});
}
