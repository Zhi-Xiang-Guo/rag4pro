/* Adapted from Nageoffer Ragent frontend/src/components/document/PdfPreview.tsx under Apache-2.0. */
import { useEffect, useRef, useState } from 'react';
import { getDocument, GlobalWorkerOptions, type PDFDocumentProxy, type RenderTask } from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import type { ApiClient } from '../api';

GlobalWorkerOptions.workerSrc = workerUrl;

export default function PdfPreview({api, path, page}: {api: ApiClient; path: string; page: number}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const [pdf, setPdf] = useState<PDFDocumentProxy>();
  const [width, setWidth] = useState(0);
  const [pageNumber, setPageNumber] = useState(page);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    let task: ReturnType<typeof getDocument> | undefined;
    setPdf(undefined);
    setError('');
    void (async () => {
      try {
        const response = await api.request(path, {signal: controller.signal});
        const data = await response.arrayBuffer();
        if (controller.signal.aborted) return;
        task = getDocument({data, isEvalSupported: false, useSystemFonts: true});
        const loaded = await task.promise;
        if (!controller.signal.aborted) setPdf(loaded);
      } catch {
        if (!controller.signal.aborted) setError('PDF 预览失败，请尝试下载原件。');
      }
    })();
    return () => {
      controller.abort();
      if (task) void task.destroy().catch(() => undefined);
    };
  }, [api, path]);

  useEffect(() => setPageNumber(page), [page, path]);
  useEffect(() => {
    const host = hostRef.current!;
    const observer = new ResizeObserver(() => setWidth(host.clientWidth));
    observer.observe(host);
    return () => observer.disconnect();
  }, []);
  const selected = pdf ? Math.min(pageNumber, pdf.numPages) : pageNumber;

  useEffect(() => {
    if (!pdf || width < 10) return;
    let active = true;
    let task: RenderTask | undefined;
    void (async () => {
      try {
        const current = await pdf.getPage(selected);
        if (!active) return;
        const canvas = canvasRef.current!;
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        const scale = Math.min(width, 920) / current.getViewport({scale: 1}).width;
        const viewport = current.getViewport({scale: scale * ratio});
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        canvas.style.width = viewport.width / ratio + 'px';
        canvas.style.height = viewport.height / ratio + 'px';
        const context = canvas.getContext('2d');
        if (!context) return;
        task = current.render({canvasContext: context, viewport});
        await task.promise;
      } catch (error) {
        if (active && !(error instanceof Error && error.name === 'RenderingCancelledException')) setError('PDF 页面绘制失败。');
      }
    })();
    return () => {
      active = false;
      task?.cancel();
    };
  }, [pdf, selected, width]);

  return <div className="file-preview" ref={hostRef}>
    {error ? <p role="alert" className="error">{error}</p> : !pdf && <p className="muted">正在读取 PDF…</p>}
    {pdf && <div className="actions preview-toolbar">
      <button disabled={selected <= 1} onClick={() => setPageNumber(selected - 1)}>上一页</button>
      <span>第 {selected} / {pdf.numPages} 页</span>
      <button disabled={selected >= pdf.numPages} onClick={() => setPageNumber(selected + 1)}>下一页</button>
    </div>}
    <canvas ref={canvasRef} hidden={!pdf} aria-label={'PDF 第 ' + selected + ' 页'} />
  </div>;
}
