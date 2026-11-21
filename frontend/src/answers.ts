import { withQuery, type ApiClient } from './api';
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
  const reader = response.body?.getReader();
  if (!reader) throw new Error('流式响应没有正文。');
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      let end = buffer.indexOf('\n\n');
      while (end >= 0) {
        const packet = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        const lines = packet.split('\n');
        const type = lines.find(line => line.startsWith('event:'))?.slice(6).trim();
        const data = lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
        if (data) {
          const payload = JSON.parse(data);
          if (type === 'delta') onDelta(payload.text);
          if (type === 'result') return payload as Answer;
          if (type === 'error') throw new Error(payload.error || '流式回答失败');
        }
        end = buffer.indexOf('\n\n');
      }
      if (done) throw new Error('流式连接在返回正式答案前断开。');
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
