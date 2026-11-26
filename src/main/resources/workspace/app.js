'use strict';

const $ = id => document.getElementById(id);
const strategies = {
    standard: '原版混合', lexical: 'BM25', dense: '向量', rrf: 'BM25 + 向量 RRF',
    rrf_diverse: 'RRF · 文档多样性'
};
const contexts = {leaf: '命中块', parent: '父段补齐', section: '章节补齐'};
const chunking = {
    fixed: ['固定窗口', '按 token 窗口切分，可设置重叠。'],
    section: ['结构分块', '跟随 Markdown 标题和段落，在句边界内切分。'],
    parent_child: ['父子分块', '索引子块，检索后可补回相邻子块组成的父段。'],
    hierarchical: ['多层分块', '索引子块，同时保存父段和章节窗口供回答使用。'],
    general: ['原版分块', '使用项目原有的 RAGFlow 文本分隔与合并策略。']
};
const pages = {chat: '对话', library: '知识库', retrieve: '检索实验'};
const traceNames = {
    lexical_independent: 'BM25 独立召回', dense_independent: '向量独立召回',
    independent_rank: '排名融合', initial: '首次召回', relaxed: '词法放宽',
    dense_fallback: '向量回退', fulltext: '全文召回', rerank: '二阶段排序'
};
const historyKey = 'rag4pro.conversations.v1';

function loadHistory() {
    try {
        const stored = JSON.parse(localStorage.getItem(historyKey));
        if (Array.isArray(stored?.sessions)) return stored;
    } catch {
        // A new conversation remains available when local storage is disabled.
    }
    return {sessions: [], currentId: ''};
}

const state = {
    ...loadHistory(), page: 'chat', datasets: [], demo: null, health: null, datasetId: '',
    token: '', chatBusy: false, inspection: 'evidence', inspectedId: '', importDataset: '',
    editingDataset: '', documentsLoad: 0
};

function persistHistory() {
    try {
        localStorage.setItem(historyKey, JSON.stringify({sessions: state.sessions.slice(0, 16), currentId: state.currentId}));
    } catch {
        // Keep the current session usable in memory if the browser quota is full.
    }
}

function element(tag, style, text) {
    const node = document.createElement(tag);
    if (style) node.className = style;
    if (text != null) node.textContent = text;
    return node;
}

function button(label, style, action) {
    const node = element('button', style, label);
    node.type = 'button';
    node.addEventListener('click', action);
    return node;
}

function selectOptions(id, options, selected) {
    const select = $(id);
    select.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
    if (options.some(([value]) => value === selected)) select.value = selected;
}

function notify(message, isError = false) {
    const notice = $('notice');
    notice.textContent = message || '';
    notice.hidden = !message;
    notice.classList.toggle('is-error', isError);
}

function formError(id, error) {
    $(id).hidden = !error;
    $(id).textContent = error?.message || '';
}

async function api(path, method = 'GET', data) {
    const headers = new Headers();
    if (state.token) headers.set('Authorization', 'Bearer ' + state.token);
    if (data !== undefined) headers.set('Content-Type', 'application/json');
    const response = await fetch(path, {method, headers, body: data === undefined ? undefined : JSON.stringify(data)});
    const body = await response.json();
    if (!response.ok) {
        const message = response.status === 401 ? '请在连接设置中填写服务访问令牌。' : (body.error || '请求失败');
        const requestId = response.headers.get('X-Request-Id');
        throw new Error(requestId ? message + '（请求 ' + requestId + '）' : message);
    }
    return body;
}

function currentConversation() {
    return state.sessions.find(session => session.id === state.currentId);
}

function datasetLabel(dataset) {
    return dataset.datasetId === state.demo?.datasetId && !dataset.registered ? '支付故障手册 · 示例' : dataset.name;
}

function renderDatasets() {
    const options = state.datasets.map(dataset => [dataset.datasetId, datasetLabel(dataset)]);
    const placeholder = options.length ? '选择知识库' : '请先新建或载入知识库';
    for (const id of ['chat-dataset', 'retrieve-dataset']) {
        selectOptions(id, [['', placeholder], ...options], state.datasetId);
        $(id).options[0].disabled = true;
    }
    $('send').disabled = state.chatBusy || !state.datasetId;
    $('datasets').replaceChildren(...state.datasets.map(dataset => {
        const item = button(datasetLabel(dataset), 'dataset-item', () => chooseDataset(dataset.datasetId));
        item.append(element('small', '', dataset.documentCount + ' 份文档'));
        item.setAttribute('aria-current', String(dataset.datasetId === state.datasetId));
        return item;
    }));
    if (!options.length) $('datasets').append(element('p', 'field-hint', '还没有知识库。'));
}

async function refreshCatalog(preferred = state.datasetId) {
    const [datasets, demo] = await Promise.all([api('/api/datasets'), api('/api/demo')]);
    state.datasets = datasets;
    state.demo = demo;
    const existing = new Set(datasets.map(dataset => dataset.datasetId));
    const choices = [preferred, currentConversation()?.datasetId, demo.datasetId, datasets[0]?.datasetId];
    state.datasetId = choices.find(id => existing.has(id)) || '';
    renderDatasets();
    renderExamples();
    if (state.page === 'library') await loadDocuments();
}

async function connect() {
    const status = $('connection-status');
    status.dataset.ready = '';
    status.textContent = '正在连接';
    try {
        state.health = await api('/api/health');
        await refreshCatalog();
        status.dataset.ready = 'yes';
        status.textContent = (state.health.index?.engine || '向量库') + ' 已连接';
        $('connection-details').textContent = '回答模型 ' + state.health.chatModel + ' · 索引 ' + state.health.index.index;
    } catch (error) {
        status.dataset.ready = 'no';
        status.textContent = '连接异常';
        throw error;
    }
}

function setPage(page) {
    if (!pages[page]) return;
    state.page = page;
    $('page-title').textContent = pages[page];
    for (const name of Object.keys(pages)) $(name + '-page').hidden = page !== name;
    for (const nav of document.querySelectorAll('[data-page]')) {
        if (nav.dataset.page === page) nav.setAttribute('aria-current', 'page');
        else nav.removeAttribute('aria-current');
    }
    if (page === 'library') loadDocuments().catch(error => notify(error.message, true));
}

function chooseDataset(id) {
    if (state.page === 'chat' && currentConversation()?.datasetId !== id) {
        state.currentId = '';
        state.inspectedId = '';
        persistHistory();
        renderChat();
    }
    state.datasetId = id;
    renderDatasets();
    if (state.page === 'library') loadDocuments().catch(error => notify(error.message, true));
}

function newChat() {
    state.currentId = '';
    state.inspectedId = '';
    $('question').value = '';
    persistHistory();
    setPage('chat');
    renderChat();
    $('question').focus();
}

function renderExamples() {
    const examples = $('examples');
    examples.replaceChildren();
    for (const example of state.demo?.examples?.slice(0, 4) || []) {
        const item = button('', 'example', () => useExample(example));
        item.disabled = state.chatBusy;
        item.append(element('strong', '', example.label), element('small', '', example.question));
        examples.append(item);
    }
    if (state.demo && !state.demo.examples?.length)
        examples.append(button('载入支付手册示例', 'secondary', () => seedDemo().catch(error => notify(error.message, true))));
}

async function seedDemo() {
    const load = $('load-demo');
    load.disabled = true;
    try {
        state.demo = await api('/api/demo', 'POST');
        await refreshCatalog(state.demo.datasetId);
        notify('支付手册示例已建立索引。');
    } finally {
        load.disabled = false;
    }
}

async function useExample(example) {
    if (state.chatBusy) return;
    state.chatBusy = true;
    renderExamples();
    try {
        if (state.demo.indexedCount !== state.demo.documentCount) await seedDemo();
        newChat();
        chooseDataset(state.demo.datasetId);
        $('chat-strategy').value = 'rrf';
        $('question').value = example.question;
    } catch (error) {
        notify(error.message, true);
        return;
    } finally {
        state.chatBusy = false;
        renderExamples();
        renderDatasets();
    }
    await sendQuestion(example.version);
}

function sourceCard(source, number, spans = []) {
    const details = element('details', 'source-card');
    const title = source.document_name || '知识库原文';
    const prefix = number == null ? '' : '[' + number + '] ';
    details.append(element('summary', '', prefix + title));
    const level = contexts[source.context_level_kwd] || '';
    const score = typeof source.similarity === 'number' ? '得分 ' + source.similarity.toFixed(4) : '';
    const meta = [source.section_kwd, level, score].filter(Boolean).join(' · ');
    if (meta) details.append(element('span', 'source-meta', meta));
    const content = source.content || source.content_with_weight || '';
    const body = element('p');
    let offset = 0;
    for (const span of [...spans].sort((a, b) => a.start - b.start)) {
        if (span.start < offset || content.slice(span.start, span.end) !== span.text) continue;
        body.append(document.createTextNode(content.slice(offset, span.start)), element('mark', '', span.text));
        offset = span.end;
    }
    body.append(document.createTextNode(content.slice(offset)));
    details.append(body);
    return details;
}

function traceCard(step) {
    const details = element('details', 'trace-step');
    const brief = [traceNames[step.stage] || step.stage];
    if (Number.isFinite(step.returned)) brief.push(step.returned + ' 个候选');
    if (Number.isFinite(step.durationMs)) brief.push(Math.round(step.durationMs) + ' ms');
    details.append(element('summary', '', brief.join(' · ')), element('pre', '', JSON.stringify(step, null, 2)));
    return details;
}

function renderInspector(message, citationIndex) {
    const panel = $('inspector-content');
    panel.replaceChildren();
    if (!message) {
        panel.append(element('div', 'placeholder', '发送问题后查看命中的原文和检索过程。'));
        return;
    }
    if (state.inspection === 'trace') {
        panel.append(element('p', 'result-note', '检索问题　' + message.query));
        for (const step of message.trace || []) panel.append(traceCard(step));
        return;
    }
    const chunks = message.reference?.chunks || [];
    const spans = message.reference?.spans || [];
    panel.append(element('p', 'result-note', chunks.length
        ? chunks.length + ' 条引用 · ' + (message.evidenceTokens ?? 0) + ' tokens 检索上下文'
        : '本次回答没有引用。'));
    for (const [index, source] of chunks.entries()) {
        if (!source) continue;
        const card = sourceCard(source, index + 1, spans.filter(span => span.citationIndex === index));
        panel.append(card);
        if (index === citationIndex) {
            card.open = true;
            card.scrollIntoView({block: 'nearest', behavior: 'smooth'});
        }
    }
}

function selectInspection(tab, message, citationIndex) {
    state.inspection = tab;
    state.inspectedId = message?.id || '';
    for (const option of document.querySelectorAll('[data-inspect]'))
        option.setAttribute('aria-pressed', String(option.dataset.inspect === tab));
    renderInspector(message, citationIndex);
}

function latestAssistant() {
    const messages = currentConversation()?.messages || [];
    return messages.find(row => row.id === state.inspectedId) || messages.findLast(row => row.role === 'assistant');
}

function renderAnswerContent(body, message) {
    let previous = 0;
    for (const match of message.content.matchAll(/\[ID:(\d+)\]/g)) {
        body.append(document.createTextNode(message.content.slice(previous, match.index)));
        const index = Number(match[1]);
        if (message.reference?.chunks?.[index]) {
            body.append(button('[' + (index + 1) + ']', 'cite', () => selectInspection('evidence', message, index)));
        } else body.append(document.createTextNode(match[0]));
        previous = match.index + match[0].length;
    }
    body.append(document.createTextNode(message.content.slice(previous)));
}

function renderChat() {
    const conversation = currentConversation();
    const messages = conversation?.messages || [];
    const container = $('messages');
    container.replaceChildren();
    $('chat-empty').hidden = messages.length > 0;
    for (const message of messages) {
        const article = element('article', 'message ' + message.role);
        if (message.role === 'user') article.textContent = message.content;
        else {
            const heading = element('div', 'message-head');
            heading.append(element('strong', '', 'rag4pro'));
            if (message.model) heading.append(element('span', '', message.model));
            const body = element('div', 'message-body');
            renderAnswerContent(body, message);
            article.append(heading, body);
        }
        container.append(article);
    }
    if (state.chatBusy && messages.at(-1)?.role === 'user')
        container.append(element('p', 'pending', '正在检索和生成回答…'));
    $('chat-scroll').scrollTop = $('chat-scroll').scrollHeight;
    $('send').disabled = state.chatBusy || !state.datasetId;
    $('send').textContent = state.chatBusy ? '处理中' : '发送';
    $('chat-dataset').disabled = state.chatBusy;
    $('sessions').replaceChildren(...state.sessions.map(session => {
        const item = button(session.title, 'session-item', () => {
            if (state.chatBusy) return;
            state.currentId = session.id;
            state.inspectedId = '';
            state.datasetId = state.datasets.some(dataset => dataset.datasetId === session.datasetId) ? session.datasetId : '';
            renderDatasets();
            persistHistory();
            setPage('chat');
            renderChat();
        });
        item.setAttribute('aria-current', String(session.id === state.currentId));
        return item;
    }));
    renderInspector(latestAssistant());
}

async function sendQuestion(version) {
    const question = $('question').value.trim();
    if (!question || state.chatBusy || !state.datasetId) return;
    let session = currentConversation();
    if (!session || session.datasetId !== state.datasetId) {
        session = {
            id: crypto.randomUUID(), datasetId: state.datasetId, version,
            title: question.slice(0, 38), messages: []
        };
        state.sessions.unshift(session);
        state.sessions = state.sessions.slice(0, 16);
        state.currentId = session.id;
    }
    const user = {id: crypto.randomUUID(), role: 'user', content: question};
    session.messages.push(user);
    const input = {
        datasetId: session.datasetId, answerMode: $('chat-answer').value, strategy: $('chat-strategy').value,
        contextMode: $('chat-context').value, evidenceBudget: 1800, topK: 8, candidateK: 64, refineMultiturn: true,
        emptyResponse: '知识库里没有找到能支持回答的内容。',
        messages: session.messages.slice(-12).map(({role, content}) => ({role, content}))
    };
    if (session.version) input.metadataFilter = {
        method: 'manual', manual: [{key: 'version', op: '=', value: session.version}]
    };
    state.chatBusy = true;
    state.inspectedId = '';
    $('question').value = '';
    notify('');
    renderChat();
    try {
        const answer = await api('/api/chat', 'POST', input);
        session.messages.push({
            id: crypto.randomUUID(), role: 'assistant', content: answer.answer,
            reference: answer.reference, query: answer.query, model: answer.model,
            trace: answer.retrievalTrace, evidenceTokens: answer.evidenceTokens
        });
        if (answer.warnings?.length) notify(answer.warnings.join(' · '));
    } catch (error) {
        session.messages = session.messages.filter(message => message !== user);
        $('question').value = question;
        notify(error.message, true);
    } finally {
        state.chatBusy = false;
        persistHistory();
        renderChat();
        renderExamples();
    }
}

async function loadDocuments() {
    const id = state.datasetId;
    const dataset = state.datasets.find(item => item.datasetId === id);
    $('library-name').textContent = dataset ? datasetLabel(dataset) : '选择知识库';
    $('library-config').textContent = dataset
        ? [dataset.documentCount + ' 份文档', chunking[dataset.chunkStrategy || 'general']?.[0],
            dataset.chunkTokenSize ? dataset.chunkTokenSize + ' tokens' : ''].filter(Boolean).join(' · ') : '';
    $('edit-dataset').hidden = !dataset;
    $('delete-dataset').hidden = !dataset?.registered || dataset.documentCount !== 0;
    $('open-import').hidden = !dataset;
    const container = $('documents');
    const load = ++state.documentsLoad;
    container.replaceChildren(element('div', 'placeholder', dataset ? '正在读取文档…' : '新建知识库后可导入文档。'));
    if (!dataset) return;
    const docs = await api('/api/documents?' + new URLSearchParams({datasetId: id}));
    if (load !== state.documentsLoad) return;
    if (!docs.length) {
        container.replaceChildren(element('div', 'placeholder', '这个知识库还没有文档。'));
        return;
    }
    const table = element('table', 'document-table');
    const header = table.createTHead().insertRow();
    for (const name of ['文档', '分块策略', '索引块', '操作']) header.append(element('th', '', name));
    const body = table.createTBody();
    for (const doc of docs) {
        const row = body.insertRow();
        row.insertCell().textContent = doc.title;
        row.insertCell().textContent = chunking[doc.chunkStrategy || 'general']?.[0] || doc.chunkStrategy;
        row.insertCell().textContent = doc.chunkCount;
        const actions = element('div', 'actions');
        actions.append(button('查看', 'secondary', () => openChunks(doc).catch(error => notify(error.message, true))),
            button('移除', 'secondary danger', () => removeDocument(doc).catch(error => notify(error.message, true))));
        row.insertCell().append(actions);
    }
    container.replaceChildren(table);
}

async function openChunks(document) {
    $('chunks-title').textContent = document.title;
    $('chunks-view').replaceChildren(element('p', 'result-note', '正在加载…'));
    $('chunks-dialog').showModal();
    try {
        const params = new URLSearchParams({datasetId: document.datasetId, docId: document.docId});
        const chunks = await api('/api/chunks?' + params);
        $('chunks-view').replaceChildren(element('p', 'result-note', chunks.length + ' 个索引块'));
        for (const [index, chunk] of chunks.entries()) {
            const card = sourceCard({...chunk, document_name: chunk.section_kwd || '索引块'}, index + 1);
            $('chunks-view').append(card);
        }
    } catch (error) {
        $('chunks-view').replaceChildren(element('p', 'form-error', error.message));
    }
}

async function removeDocument(doc) {
    if (!confirm('移除「' + doc.title + '」及其索引块？')) return;
    await api('/api/documents?' + new URLSearchParams({datasetId: doc.datasetId, docId: doc.docId}), 'DELETE');
    await refreshCatalog(doc.datasetId);
    notify('文档已移除。');
}

function applyChunkOptions(prefix, options = {}) {
    $(prefix + '-strategy').value = options.chunkStrategy || 'parent_child';
    $(prefix + '-size').value = Math.max(32, options.chunkTokenSize || 160);
    $(prefix + '-overlap').value = options.overlapPercent || 0;
    updateChunkDescription(prefix);
}

function updateChunkDescription(prefix) {
    const strategy = $(prefix + '-strategy').value;
    $(prefix + '-description').textContent = chunking[strategy]?.[1] || '';
    $(prefix + '-overlap-label').hidden = strategy !== 'fixed';
}

function chunkOptions(prefix) {
    const chunkStrategy = $(prefix + '-strategy').value;
    return {
        chunkStrategy, chunkTokenSize: Number($(prefix + '-size').value),
        overlapPercent: chunkStrategy === 'fixed' ? Number($(prefix + '-overlap').value) : 0
    };
}

function openDataset(dataset) {
    state.editingDataset = dataset?.datasetId || '';
    $('dataset-title').textContent = dataset ? '知识库设置' : '新建知识库';
    $('dataset-name').value = dataset ? datasetLabel(dataset) : '';
    $('dataset-language').value = dataset?.language || '';
    applyChunkOptions('dataset', dataset);
    formError('dataset-error');
    $('dataset-dialog').showModal();
}

async function saveDataset(event) {
    event.preventDefault();
    const submit = event.submitter;
    const name = $('dataset-name').value.trim();
    if (!name) return;
    if (submit) submit.disabled = true;
    formError('dataset-error');
    try {
        const datasetId = state.editingDataset || 'kb-' + crypto.randomUUID();
        await api('/api/datasets', 'POST', {datasetId, name, language: $('dataset-language').value, ...chunkOptions('dataset')});
        $('dataset-dialog').close();
        await refreshCatalog(datasetId);
        notify('知识库已保存。');
    } catch (error) {
        formError('dataset-error', error);
    } finally {
        if (submit) submit.disabled = false;
    }
}

function openImport() {
    const dataset = state.datasets.find(item => item.datasetId === state.datasetId);
    if (!dataset) return;
    state.importDataset = dataset.datasetId;
    $('import-form').reset();
    $('import-title').textContent = '导入到 ' + datasetLabel(dataset);
    $('chunk-preview').hidden = true;
    applyChunkOptions('document', dataset);
    formError('import-error');
    $('import-dialog').showModal();
}

function documentInput() {
    return {
        datasetId: state.importDataset, title: $('document-title').value.trim(), text: $('document-text').value,
        fileType: $('document-format').value, ...chunkOptions('document')
    };
}

async function previewChunks() {
    if (!$('import-form').reportValidity()) return;
    $('preview-chunks').disabled = true;
    formError('import-error');
    try {
        const result = await api('/api/chunk-preview', 'POST', documentInput());
        const preview = $('chunk-preview');
        const counts = [result.total + ' 个检索块', result.parentCount + ' 个父段', result.sectionCount + ' 个章节窗口'];
        preview.replaceChildren(element('p', 'result-note', counts.join(' · ')));
        for (const [index, piece] of result.pieces.entries()) {
            const card = sourceCard({document_name: piece.section || '索引块', content: piece.text,
                section_kwd: piece.tokens + ' tokens'}, index + 1);
            preview.append(card);
        }
        preview.hidden = false;
    } catch (error) {
        formError('import-error', error);
    } finally {
        $('preview-chunks').disabled = false;
    }
}

async function importDocument(event) {
    event.preventDefault();
    $('import-document').disabled = true;
    formError('import-error');
    try {
        const result = await api('/api/documents', 'POST', {...documentInput(), docId: crypto.randomUUID()});
        $('import-dialog').close();
        await refreshCatalog(state.importDataset);
        notify('索引已建立。' + result.chunkCount + ' 个检索块，' + result.parentCount + ' 个上下文块。');
    } catch (error) {
        formError('import-error', error);
    } finally {
        $('import-document').disabled = false;
    }
}

function renderComparison(result, label, strategy) {
    const card = element('article', 'panel compare-result');
    card.append(element('h3', '', label + ' · ' + strategies[strategy] + ' · ' + contexts[result.contextMode]));
    const metrics = element('dl', 'metrics');
    for (const [name, value] of [['返回命中', result.hits.length], ['证据 tokens', result.evidenceTokens], ['召回耗时', Math.round(result.retrievalMs) + ' ms']]) {
        const cell = element('div');
        cell.append(element('dt', '', name), element('dd', '', value));
        metrics.append(cell);
    }
    card.append(metrics, element('h4', '', '召回命中'));
    if (!result.hits.length) card.append(element('p', 'result-note', '未召回文档块。'));
    for (const [index, hit] of result.hits.entries()) card.append(sourceCard(hit, index + 1));
    card.append(element('h4', '', '预算内上下文'));
    if (!result.evidence.length) card.append(element('p', 'result-note', '没有可用证据。'));
    for (const source of result.evidence) card.append(sourceCard(source));
    card.append(element('h4', '', '检索轨迹'));
    for (const stage of result.trace) card.append(traceCard(stage));
    return card;
}

async function compareRetrieval(event) {
    event.preventDefault();
    const question = $('retrieve-question').value.trim();
    if (!question || !state.datasetId) return;
    const base = {datasetId: state.datasetId, question, topK: Number($('top-k').value),
        candidateK: Number($('candidate-k').value), evidenceBudget: Number($('evidence-budget').value)};
    if (base.candidateK < base.topK) {
        notify('候选池须不小于最终 Top K。', true);
        return;
    }
    notify('');
    $('run-retrieve').disabled = true;
    $('comparison').replaceChildren();
    try {
        for (const [prefix, label] of [['baseline', '基线'], ['compare', '对照']]) {
            const strategy = $(prefix + '-strategy').value;
            const result = await api('/api/retrieve/context', 'POST', {
                ...base, strategy, contextMode: $(prefix + '-context').value
            });
            $('comparison').append(renderComparison(result, label, strategy));
        }
    } catch (error) {
        notify(error.message, true);
    } finally {
        $('run-retrieve').disabled = false;
    }
}

for (const id of ['chat-strategy', 'baseline-strategy', 'compare-strategy'])
    selectOptions(id, Object.entries(strategies), id === 'baseline-strategy' ? 'standard' : 'rrf');
for (const id of ['chat-context', 'baseline-context', 'compare-context'])
    selectOptions(id, Object.entries(contexts), id === 'baseline-context' ? 'leaf' : 'parent');
for (const prefix of ['dataset', 'document']) {
    selectOptions(prefix + '-strategy', Object.entries(chunking).map(([id, value]) => [id, value[0]]), 'parent_child');
    $(prefix + '-strategy').addEventListener('change', () => updateChunkDescription(prefix));
}
for (const label of document.querySelectorAll('[data-page]'))
    label.addEventListener('click', () => { notify(''); setPage(label.dataset.page); });
for (const tab of document.querySelectorAll('[data-inspect]'))
    tab.addEventListener('click', () => selectInspection(tab.dataset.inspect, latestAssistant()));
for (const close of document.querySelectorAll('[data-close]'))
    close.addEventListener('click', () => close.closest('dialog').close());
for (const id of ['chat-dataset', 'retrieve-dataset'])
    $(id).addEventListener('change', event => chooseDataset(event.target.value));

$('new-chat').addEventListener('click', newChat);
$('chat-form').addEventListener('submit', event => { event.preventDefault(); void sendQuestion(); });
$('question').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        $('chat-form').requestSubmit();
    }
});
$('load-demo').addEventListener('click', () => seedDemo().catch(error => notify(error.message, true)));
$('create-dataset').addEventListener('click', () => openDataset());
$('edit-dataset').addEventListener('click', () => openDataset(state.datasets.find(item => item.datasetId === state.datasetId)));
$('delete-dataset').addEventListener('click', async () => {
    const id = state.datasetId;
    if (!confirm('删除这个空知识库？')) return;
    try {
        await api('/api/datasets?' + new URLSearchParams({datasetId: id}), 'DELETE');
        await refreshCatalog('');
        notify('空知识库已删除。');
    } catch (error) { notify(error.message, true); }
});
$('dataset-form').addEventListener('submit', saveDataset);
$('open-import').addEventListener('click', openImport);
$('preview-chunks').addEventListener('click', previewChunks);
$('import-form').addEventListener('submit', importDocument);
$('retrieve-form').addEventListener('submit', compareRetrieval);
$('document-file').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > 1_600_000 || !/\.(md|txt)$/i.test(file.name)) {
        formError('import-error', new Error('请选择小于 1.6 MB 的 TXT 或 Markdown 文件。'));
        return;
    }
    formError('import-error');
    try {
        $('document-title').value = file.name;
        $('document-format').value = /\.md$/i.test(file.name) ? 'md' : 'txt';
        $('document-text').value = await file.text();
    } catch (error) { formError('import-error', error); }
});
$('open-connection').addEventListener('click', () => $('connection-dialog').showModal());
$('connection-form').addEventListener('submit', async event => {
    event.preventDefault();
    state.token = $('connection-token').value.trim();
    formError('connection-error');
    try {
        await connect();
        $('connection-dialog').close();
        notify('');
    } catch (error) { formError('connection-error', error); }
});
renderDatasets();
renderChat();
connect().catch(error => notify(error.message, true));
