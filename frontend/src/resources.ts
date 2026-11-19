import { useEffect, useState } from 'react';
import { messageOf, withQuery, type ApiClient } from './api';

export type DocumentRow = {
  datasetId: string; docId: string; title: string; chunkCount: number;
  sourceRevision?: string; sourceName?: string; sourceFormat?: string;
  chunkStrategy?: string; chunkTokenSize?: number; overlapPercent?: number;
};
export type JobStatus = 'QUEUED' | 'PARSING' | 'INDEXING' | 'READY' | 'FAILED';
export type Job = {
  id: string; docId: string; title: string; filename: string;
  status: JobStatus; updatedAt: string; error: string; expectedVersion: string;
};
export type SourceLocation = {revision: string; kind: string; unit: number; start?: number};
export type SourcePointer = {
  datasetId: string; docId: string; revision?: string;
  location?: SourceLocation; content?: string; span?: {start: number; end: number; text: string};
};
export type SourceUnit = {kind: string; number: number; text: string};
export type SourceVersion = {
  revision: string; name: string; format: 'txt' | 'md' | 'pdf' | 'docx'; sha256: string; units: SourceUnit[];
};
export type IndexChunk = {
  id: string; content_with_weight: string; available_int?: number; chunk_order_int?: number;
  source_revision_kwd?: string; source_kind_kwd?: string; source_unit_int?: number; source_start_int?: number;
  mom_id?: string; section_id?: string; section_kwd?: string; context_level_kwd?: string;
};
export const chunkStrategies: Record<string, string> = {
  general: '原版分块', fixed: '固定窗口', section: '结构分块',
  parent_child: '父子分块', hierarchical: '多层分块',
};
export const jobLabels: Record<JobStatus, string> = {
  QUEUED: '排队中', PARSING: '提取正文', INDEXING: '建立索引', READY: '可检索', FAILED: '入库失败',
};
export const isPending = (job?: Job) => !!job && ['QUEUED', 'PARSING', 'INDEXING'].includes(job.status);
export const knowledgeUrl = (datasetId: string) => '/knowledge/' + encodeURIComponent(datasetId);
export const documentUrl = (datasetId: string, docId: string) => knowledgeUrl(datasetId) + '/documents/' + encodeURIComponent(docId);
export const sourceUrl = (source: SourcePointer, file = false) => withQuery('/api/document-source' + (file ? '/file' : ''), {
  datasetId: source.datasetId, docId: source.docId, revision: source.revision || source.location?.revision,
});

export function useResource<T>(api: ApiClient, path?: string, refresh = 0) {
  const [state, setState] = useState<{path?: string; data?: T; error: string}>({error: ''});
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    api.json<T>(path, 'GET', undefined, controller.signal).then(data => {
      if (!controller.signal.aborted) setState({path, data, error: ''});
    }).catch(error => {
      if (!controller.signal.aborted) setState({path, error: messageOf(error)});
    });
    return () => controller.abort();
  }, [api, path, refresh]);
  if (state.path !== path) return {data: undefined, error: '', loading: !!path};
  return {...state, loading: !!path && !state.data && !state.error};
}

export function useDocuments(api: ApiClient, datasetId: string, refresh: number) {
  const [result, setResult] = useState<{datasetId?: string; documents: DocumentRow[]; jobs: Job[]; error: string}>({
    documents: [], jobs: [], error: '',
  });
  useEffect(() => {
    const controller = new AbortController();
    let timer = 0;
    const fetchData = async () => {
      try {
        const query = {datasetId};
        const [documents, jobs] = await Promise.all([
          api.json<DocumentRow[]>(withQuery('/api/documents', query), 'GET', undefined, controller.signal),
          api.json<Job[]>(withQuery('/api/ingestion', query), 'GET', undefined, controller.signal),
        ]);
        if (controller.signal.aborted) return;
        setResult({datasetId, documents, jobs, error: ''});
        if (jobs.some(isPending)) timer = window.setTimeout(fetchData, 1200);
      } catch (error) {
        if (!controller.signal.aborted) setResult(old => ({...old, datasetId, error: messageOf(error)}));
      }
    };
    void fetchData();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [api, datasetId, refresh]);
  return result.datasetId === datasetId ? {...result, loading: false}
    : {documents: [], jobs: [], error: '', loading: true};
}
