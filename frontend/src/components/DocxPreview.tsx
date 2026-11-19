/* Adapted from Nageoffer Ragent frontend/src/components/document/DocxPreview.tsx under Apache-2.0. */
import { useEffect, useRef, useState } from 'react';
import { renderAsync } from 'docx-preview';
import type { ApiClient } from '../api';
import type { SourceUnit } from '../resources';

export default function DocxPreview({api, path, unit}: {api: ApiClient; path: string; unit?: SourceUnit}) {
  const host = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<'loading' | 'done' | 'error'>('loading');

  useEffect(() => {
    const controller = new AbortController();
    const element = host.current!;
    element.replaceChildren();
    setStatus('loading');
    void (async () => {
      try {
        const response = await api.request(path, {signal: controller.signal});
        const bytes = await response.arrayBuffer();
        if (controller.signal.aborted) return;
        const detached = document.createElement('div');
        await renderAsync(bytes, detached, detached, {
          className: 'docx', inWrapper: true, breakPages: false, ignoreWidth: true, ignoreHeight: true,
          renderHeaders: false, renderFooters: false, renderFootnotes: false, renderEndnotes: false,
          renderAltChunks: false, useBase64URL: true,
        });
        if (controller.signal.aborted) return;
        detached.querySelectorAll('a').forEach(anchor => anchor.removeAttribute('href'));
        element.replaceChildren(...detached.childNodes);
        setStatus('done');
      } catch {
        if (!controller.signal.aborted) setStatus('error');
      }
    })();
    return () => { controller.abort(); element.replaceChildren(); };
  }, [api, path]);

  useEffect(() => {
    const element = host.current;
    if (!element || status !== 'done') return;
    element.querySelectorAll('.anchored-block').forEach(node => node.classList.remove('anchored-block'));
    if (!unit || !['paragraph', 'table'].includes(unit.kind)) return;
    const nodes = element.querySelectorAll(unit.kind === 'table' ? 'article > table' : 'article > p');
    const target = nodes[unit.number - 1];
    // Pagination and formatting can change the visual DOM. Only select an equivalent block.
    const normalize = (value: string) => value.replace(/\s+/g, '');
    if (target && normalize(target.textContent ?? '') === normalize(unit.text)) {
      target.classList.add('anchored-block');
      target.scrollIntoView({block: 'nearest'});
    }
  }, [status, unit]);

  return <div className="file-preview word-preview">
    {status === 'loading' && <p className="muted">正在读取 Word…</p>}
    {status === 'error' && <p role="alert" className="error">Word 预览失败，请尝试下载原件。</p>}
    <div ref={host} />
  </div>;
}
