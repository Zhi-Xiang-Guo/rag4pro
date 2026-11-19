import { useEffect, useMemo, useState, type FormEvent } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Link, Navigate, NavLink, Route, Routes } from 'react-router-dom';
import { ArrowUpRight, BookOpen, ChartNoAxesColumnIncreasing, Database, KeyRound, MessageSquare, ScanText } from 'lucide-react';
import { apiClient, savedToken, saveToken, type ServiceStatus } from './api';
import Knowledge from './Knowledge';
import './styles.css';

const pages = [
  {to: '/chat', title: '对话', icon: MessageSquare},
  {to: '/knowledge', title: '知识库', icon: Database},
  {to: '/retrieve', title: '检索实验', icon: ChartNoAxesColumnIncreasing},
];

function Workspace() {
  const [token, setToken] = useState(savedToken);
  const [connection, setConnection] = useState(false);
  const [health, setHealth] = useState<ServiceStatus>();
  const api = useMemo(() => apiClient(token), [token]);

  useEffect(() => {
    const controller = new AbortController();
    api.json<ServiceStatus>('/api/health', 'GET', undefined, controller.signal).then(setHealth).catch(() => {
      if (!controller.signal.aborted) setHealth(undefined);
    });
    return () => controller.abort();
  }, [api]);

  function connect(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const next = String(new FormData(event.currentTarget).get('token') ?? '').trim();
    saveToken(next);
    setToken(next);
    setConnection(false);
  }

  return (
    <div className="shell">
      <aside className="sidebar" aria-label="工作台导航">
        <Link to="/chat" className="brand"><span className="brand-mark"><ScanText size={20} /></span><strong>rag4pro</strong></Link>
        <nav>{pages.map(({to, title, icon: Icon}) => (
          <NavLink key={to} to={to}><Icon size={18} />{title}</NavLink>
        ))}</nav>
        <div className="sidebar-end">
          <a href="/demo"><BookOpen size={18} /> 支付手册示例 <ArrowUpRight size={13} /></a>
          <button type="button" className="nav-button" onClick={() => setConnection(true)}><KeyRound size={18} /> 连接设置</button>
          {health?.chatModel && <p className="model-name" title={health.chatModel}>{health.chatModel}</p>}
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/knowledge/*" element={<Knowledge key={token} api={api} />} />
          <Route path="/chat/*" element={<section className="page"><h1>对话</h1><a href="/workspace-legacy">打开对话工作台</a></section>} />
          <Route path="/retrieve" element={<section className="page"><h1>检索实验</h1><a href="/workspace-legacy">打开检索实验</a></section>} />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </main>
      {connection && (
        <div className="overlay" onClick={() => setConnection(false)}>
          <form className="dialog" role="dialog" aria-modal="true" aria-label="连接设置" onClick={e => e.stopPropagation()} onSubmit={connect}>
            <h2>连接设置</h2>
            <label>服务令牌<input autoFocus name="token" type="password" defaultValue={token} autoComplete="off" /></label>
            <div className="actions"><button type="button" onClick={() => setConnection(false)}>取消</button><button className="primary">保存</button></div>
          </form>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<BrowserRouter><Workspace /></BrowserRouter>);
