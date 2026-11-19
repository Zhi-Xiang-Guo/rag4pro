import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { CircleStop, Plus, SendHorizontal, MessageSquare, ScanText } from 'lucide-react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { citationMarkdown } from './citationMarkdown';
import { messageOf, withQuery, type ApiClient, type Dataset } from './api';
import { contexts, strategies, streamAnswer, type Answer } from './answers';
import { AnswerSources, EvidenceDialog, type SelectedSource } from './components/AnswerSources';
import { useResource } from './resources';
import { decodeHistory, historyKey, recentHistory, type Message, type Session, type History } from './conversations';

type Running = {streamId: string; controller: AbortController; streaming: boolean; target: Session; message: Message};
function loadHistory(): History {
  try { return decodeHistory(localStorage.getItem(historyKey)); }
  catch { return decodeHistory(null); }
}

function cancel(api: ApiClient, run: Running) {
  if (run.streaming) {
    void api.request(withQuery('/api/chat/stream', {streamId: run.streamId}), {method: 'DELETE'}).catch(() => undefined);
  }
  run.controller.abort();
}

function AssistantMessage({message, onCitation}: {message: Message; onCitation: (index: number) => void}) {
  return <article className="chat-message assistant">
    <header><span className="assistant-icon"><ScanText size={14} /></span><strong>rag4pro</strong>
      {message.model && <span className="muted">{message.model}</span>}</header>
    <div className="markdown"><Markdown skipHtml remarkPlugins={[remarkGfm, citationMarkdown(message.reference?.chunks?.length ?? 0)]}
      components={{
        span: ({node, children}) => {
          const marker = node?.properties['data-citation-index'];
          const index = typeof marker === 'string' ? Number(marker) : -1;
          if (message.reference?.chunks?.[index]) {
            return <button type="button" className="citation" aria-label={'查看引用 ' + (index + 1)}
              onClick={() => onCitation(index)}>{children}</button>;
          }
          return <span>{children}</span>;
        },
        a: ({href, children}) => <a href={href} rel="noreferrer">{children}</a>,
        img: ({alt}) => alt ? <span>{alt}</span> : null,
      }}>{message.content}</Markdown></div>
  </article>;
}

export default function Chat({api}: {api: ApiClient}) {
  const datasets = useResource<Dataset[]>(api, '/api/datasets');
  const [params, setParams] = useSearchParams();
  const [history, setHistory] = useState(loadHistory);
  const [question, setQuestion] = useState('');
  const [answerMode, setAnswerMode] = useState('standard');
  const [strategy, setStrategy] = useState('rrf');
  const [contextMode, setContextMode] = useState('parent');
  const [pending, setPending] = useState<Running>();
  const [live, setLive] = useState('');
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');
  const [inspection, setInspection] = useState<{messageId: string; index?: number}>();
  const [reading, setReading] = useState<SelectedSource>();
  const active = useRef<Running>();
  const messageList = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const saved = history.sessions.find(session => session.id === history.currentId);
  const requestedDataset = params.get('datasetId');
  const datasetId = requestedDataset || saved?.datasetId || datasets.data?.[0]?.datasetId || '';
  const docId = requestedDataset ? params.get('docId') ?? undefined : saved?.docId;
  const session = saved?.datasetId === datasetId && saved.docId === docId ? saved : undefined;
  const messages = session?.messages ?? [];
  const busy = !!pending;
  const showingPending = pending && pending.target.datasetId === datasetId && pending.target.docId === docId;
  const visibleMessages = showingPending ? [...messages, pending.message] : messages;
  const assistant = messages.find(message => message.id === inspection?.messageId)
    ?? [...messages].reverse().find(message => message.role === 'assistant');
  const selectedIndex = assistant?.id === inspection?.messageId ? inspection?.index : undefined;

  useEffect(() => {
    try { localStorage.setItem(historyKey, JSON.stringify(history)); }
    catch { /* Keep the current session available in memory. */ }
  }, [history]);
  useEffect(() => () => {
    const run = active.current;
    if (run) cancel(api, run);
  }, [api]);
  useEffect(() => {
    const area = messageList.current;
    if (area) area.scrollTop = area.scrollHeight;
  }, [messages, pending, live]);

  function newChat(nextDataset = datasetId, nextDocument = docId) {
    setHistory(old => ({...old, currentId: ''}));
    setParams(nextDataset ? {datasetId: nextDataset, ...(nextDocument ? {docId: nextDocument} : {})} : {});
    setQuestion(''); setError(''); setNotice(''); setInspection(undefined);
    input.current?.focus();
  }

  function openSession(target: Session) {
    setHistory(old => ({...old, currentId: target.id}));
    setParams({datasetId: target.datasetId, ...(target.docId ? {docId: target.docId} : {})});
    setQuestion(''); setError(''); setNotice(''); setInspection(undefined);
  }

  function updateMessages(target: Session) {
    setHistory(old => ({currentId: target.id, sessions: [target, ...old.sessions.filter(item => item.id !== target.id)].slice(0, 16)}));
  }

  function stop() {
    const run = active.current;
    if (!run) return;
    active.current = undefined;
    cancel(api, run);
    setQuestion(draft => draft || run.message.content);
    setLive(''); setPending(undefined); setNotice('已停止生成。');
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = question.trim();
    if (!text || !datasetId || active.current) return;
    const target: Session = session ?? {id: crypto.randomUUID(), title: text.slice(0, 38), datasetId, docId, messages: []};
    const user: Message = {id: crypto.randomUUID(), role: 'user', content: text};
    const request = {
      datasetId, docId, answerMode, strategy, contextMode, evidenceBudget: 1800, topK: 8, candidateK: 64,
      refineMultiturn: true, emptyResponse: '知识库里没有找到能支持回答的内容。',
      messages: recentHistory(target.messages, text),
      metadataFilter: target.version ? {method: 'manual', manual: [{key: 'version', op: '=', value: target.version}]} : undefined,
    };
    const run: Running = {streamId: crypto.randomUUID(), controller: new AbortController(),
      streaming: answerMode === 'standard', target, message: user};
    active.current = run;
    setPending(run); setQuestion(''); setLive(''); setError(''); setNotice(''); setInspection(undefined);
    try {
      const result = run.streaming ? await streamAnswer(api, run.streamId, request, run.controller.signal,
        delta => { if (active.current === run) setLive(current => current + delta); })
        : await api.json<Answer>('/api/chat', 'POST', request, run.controller.signal);
      if (active.current !== run || run.controller.signal.aborted) return;
      const reply: Message = {id: crypto.randomUUID(), role: 'assistant', content: result.answer,
        reference: result.reference, model: result.model};
      updateMessages({...target, messages: [...target.messages, user, reply]});
      if (result.warnings?.length) setNotice(result.warnings.join(' · '));
    } catch (error) {
      if (active.current === run) {
        setQuestion(draft => draft || text);
        if (!run.controller.signal.aborted) setError(messageOf(error));
      }
    } finally {
      if (active.current === run) {
        active.current = undefined;
        setLive(''); setPending(undefined);
      }
    }
  }

  return <section className="chat-workspace">
    <aside className="session-sidebar" aria-label="最近对话">
      <button className="new-chat" disabled={busy} onClick={() => newChat()}><Plus size={16} /> 新建对话</button>
      <h2>最近对话</h2>
      {history.sessions.map(item => <button key={item.id} className={session?.id === item.id ? 'session active' : 'session'}
        type="button" disabled={busy} title={item.title} onClick={() => openSession(item)}>
        <MessageSquare size={15} /> <span>{item.title}</span>
      </button>)}
    </aside>
    <div className="chat-center">
      <header className="chat-heading"><h1>对话</h1>{docId && <span className="scope-chip">单篇文档</span>}</header>
      <div className="conversation-scroll" ref={messageList}>
        {!visibleMessages.length && <div className="welcome">
          <span className="welcome-icon"><ScanText size={24} /></span><h2>向知识库提问</h2>
          <p>用文档回答问题，点开引用核对原文。</p>
          <Link className="button" to="/knowledge">管理知识库</Link>
        </div>}
        {visibleMessages.map(message => message.role === 'user'
          ? <article key={message.id} className="chat-message user">{message.content}</article>
          : <AssistantMessage key={message.id} message={message}
              onCitation={index => {
                setInspection({messageId: message.id, index});
                const chunk = message.reference?.chunks?.[index];
                const spans = message.reference?.spans?.filter(item => item.citationIndex === index);
                if (chunk) setReading({chunk, spans, datasetId});
              }} />)}
        {showingPending && <article className="chat-message assistant" aria-live="polite">
          <header><span className="assistant-icon"><ScanText size={14} /></span><strong>rag4pro</strong></header>
          {live ? <p className="live-answer">{live}</p> : <p className="muted">正在检索并生成回答…</p>}
        </article>}
      </div>
      <form className="chat-compose" onSubmit={submit}>
        {(error || datasets.error) && <p className="error" role="alert">{error || datasets.error}</p>}
        {notice && <p className="muted" role="status">{notice}</p>}
        <div className="compose-options">
          <label>知识库<select value={datasetId} disabled={busy}
            onChange={event => { newChat(event.target.value, ''); }}>
            {!datasets.data?.length && <option value="">请选择知识库</option>}
            {datasetId && !datasets.data?.some(item => item.datasetId === datasetId) && <option value={datasetId}>{datasetId}</option>}
            {datasets.data?.map(item => <option key={item.datasetId} value={item.datasetId}>{item.name}</option>)}
          </select></label>
          <label>回答<select value={answerMode} disabled={busy} onChange={e => setAnswerMode(e.target.value)}>
            <option value="standard">综合回答</option><option value="extractive">证据摘录</option>
          </select></label>
          <label>检索<select value={strategy} disabled={busy} onChange={e => setStrategy(e.target.value)}>
            {Object.entries(strategies).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
          </select></label>
          <label>上下文<select value={contextMode} disabled={busy} onChange={e => setContextMode(e.target.value)}>
            {Object.entries(contexts).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
          </select></label>
        </div>
        <textarea ref={input} aria-label="输入问题" placeholder="输入要查询的问题…" rows={2} maxLength={2000}
          value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault(); e.currentTarget.form?.requestSubmit();
            }
          }} />
        <div className="compose-actions">
          <span className="muted">Enter 发送 · Shift + Enter 换行</span>
          {busy ? <button key="stop" type="button" onClick={event => { event.preventDefault(); stop(); }}>
              <CircleStop size={15} /> 停止</button>
            : <button key="send" type="submit" className="primary" disabled={!datasetId || !question.trim()}>
              <SendHorizontal size={15} /> 发送</button>}
        </div>
      </form>
    </div>
    <AnswerSources key={assistant?.id ?? 'empty'} api={api} reference={assistant?.reference} selectedIndex={selectedIndex} datasetId={datasetId} />
    {reading && <EvidenceDialog api={api} selected={reading} onClose={() => setReading(undefined)} />}
  </section>;
}
