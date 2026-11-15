export type Dataset = {
  datasetId: string;
  name: string;
  documentCount: number;
  registered: boolean;
  chunkStrategy?: string;
  chunkTokenSize?: number;
  overlapPercent?: number;
  language?: string;
};

export type ServiceStatus = {status: string; chatModel?: string; extractiveModel?: string; rerank?: boolean};

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

const tokenKey = 'rag4pro.workspace.token';

export function savedToken() {
  try { return sessionStorage.getItem(tokenKey) ?? ''; }
  catch { return ''; }
}

export function saveToken(token: string) {
  try {
    if (token) sessionStorage.setItem(tokenKey, token);
    else sessionStorage.removeItem(tokenKey);
  } catch { /* Keep using the token in the current tab. */ }
}

export function withQuery(path: string, values: Record<string, string | undefined>) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) if (value !== undefined) query.set(key, value);
  return query.size ? path + '?' + query : path;
}

export function apiClient(token: string) {
  async function request(path: string, init: RequestInit = {}) {
    const headers = new Headers(init.headers);
    if (token) headers.set('Authorization', 'Bearer ' + token);
    const response = await fetch(path, {...init, headers});
    if (response.ok) return response;
    const error = await response.json().catch(() => null);
    throw new ApiError(response.status, error?.error ?? '请求失败（' + response.status + '）');
  }
  async function json<T>(path: string, method = 'GET', body?: unknown, signal?: AbortSignal): Promise<T> {
    const response = await request(path, {
      method, signal,
      headers: body === undefined ? undefined : {'Content-Type': 'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return response.json();
  }
  return {request, json};
}

export type ApiClient = ReturnType<typeof apiClient>;
export const messageOf = (error: unknown) => error instanceof Error ? error.message : '请求失败';
