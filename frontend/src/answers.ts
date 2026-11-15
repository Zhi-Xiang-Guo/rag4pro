import { withQuery, type ApiClient } from './api';
import { readAnswerStream } from './answerStream';
import type { SourceLocation } from './resources';

export type Evidence = {
  id?: string; document_id?: string; document_external_id?: string; document_name?: string; dataset_id?: string;
  doc_id?: string; docnm_kwd?: string; kb_id?: string; content?: string; content_with_weight?: string;
  similarity?: number; vector_similarity?: number; term_similarity?: number; rrf_score?: number;
  section_kwd?: string; context_level_kwd?: string; source?: SourceLocation;
};
export type CitationSpan = {citationIndex: number; start: number; end: number; text: string};
export type Reference = {chunks?: Evidence[]; spans?: CitationSpan[]};
export type Answer = {
  answer: string; reference?: Reference; model?: string; warnings?: string[];
};
export const strategies: Record<string, string> = {
  standard: '原版混合', lexical: 'BM25', dense: '向量', rrf: 'BM25 + 向量 RRF', rrf_diverse: 'RRF · 文档多样性',
};
export const contexts: Record<string, string> = {leaf: '命中块', parent: '父段补齐', section: '章节补齐'};
export const excerpt = (chunk: Evidence) => chunk.content ?? chunk.content_with_weight ?? '';

export async function streamAnswer(api: ApiClient, streamId: string, input: unknown, signal: AbortSignal,
  onDelta: (value: string) => void): Promise<Answer> {
  const response = await api.request(withQuery('/api/chat/stream', {streamId}), {
    method: 'POST', headers: {'Content-Type': 'application/json', Accept: 'text/event-stream'},
    body: JSON.stringify(input), signal,
  });
  if (!response.body) throw new Error('流式响应没有正文。');
  return readAnswerStream(response.body, onDelta);
}
