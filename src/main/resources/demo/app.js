'use strict';

const $ = id => document.getElementById(id);
let catalog = null;
let busy = false;
let sourceRequest = 0;
let sourceTrigger = null;

async function api(path, {method = 'GET', payload} = {}) {
    const headers = new Headers();
    const token = $('token').value.trim();
    if (token) headers.set('Authorization', 'Bearer ' + token);
    if (payload !== undefined) headers.set('Content-Type', 'application/json');
    let response;
    try {
        response = await fetch(path, {method, headers, body: JSON.stringify(payload)});
    } catch (error) {
        throw new Error('连接中断，请检查 Java 服务。', {cause: error});
    }

    const requestId = response.headers.get('X-Request-Id');
    const body = await response.json();
    if (!response.ok) {
        const messages = {
            local_model_unavailable: '本地回答模型暂不可用，请检查 Ollama 后重试。',
            invalid_evidence_selection: '回答模型返回的引用无法校验，请重试。'
        };
        const message = response.status === 401 ? '请在连接设置中填写服务访问令牌。'
            : (messages[body.code] ?? body.error ?? '请求失败，请重试。');
        const error = new Error(requestId ? message + ' 请求 ID ' + requestId : message);
        error.status = response.status;
        throw error;
    }
    return {body, requestId};
}

function ready() {
    return catalog !== null && catalog.indexedCount === catalog.documentCount;
}

function updateControls() {
    $('ask').disabled = busy || !ready();
    $('seed').disabled = busy;
    $('connect').disabled = busy;
    for (const example of $('examples').children) example.disabled = busy || !ready();
    $('results').setAttribute('aria-busy', String(busy));
}

function showError(error) {
    $('error').textContent = error?.message ?? '';
    $('error').hidden = !error;
    if (error?.status === 401) $('connection').open = true;
}

async function perform(action) {
    if (busy) return;
    busy = true;
    showError(null);
    updateControls();
    try {
        await action();
    } catch (error) {
        showError(error);
    } finally {
        busy = false;
        updateControls();
    }
}

function button(label, onClick, className = 'secondary') {
    const element = document.createElement('button');
    element.type = 'button';
    element.className = className;
    element.textContent = label;
    element.addEventListener('click', onClick);
    return element;
}

function showCatalog(body) {
    catalog = body;
    $('catalog-status').textContent = ready() ? '示例索引就绪 · ' + body.documentCount + ' 篇文档'
        : '已索引 ' + body.indexedCount + ' / ' + body.documentCount + ' 篇示例文档';
    $('setup').hidden = ready();
    $('examples').replaceChildren();
    for (const example of body.examples) {
        $('examples').append(button(example.label, () => {
            $('version').value = example.version;
            $('question').value = example.question;
            $('question-form').requestSubmit();
        }));
    }
}

async function connect() {
    $('catalog-status').textContent = '正在检查示例索引…';
    try {
        showCatalog((await api('/api/demo')).body);
    } catch (error) {
        catalog = null;
        $('catalog-status').textContent = '示例索引状态未取得';
        $('setup').hidden = true;
        throw error;
    }
}

function hideSource(restoreFocus = false) {
    sourceRequest++;
    $('source').hidden = true;
    if (restoreFocus) sourceTrigger?.focus();
    sourceTrigger = null;
}

function highlightOriginal(text, quotes) {
    const container = $('source-text');
    container.replaceChildren();
    const ranges = [];
    for (const quote of new Set(quotes)) {
        const start = text.indexOf(quote);
        if (start !== -1) ranges.push([start, start + quote.length]);
    }
    ranges.sort((a, b) => a[0] - b[0]);
    let cursor = 0;
    for (const [start, end] of ranges) {
        if (start < cursor) continue;
        container.append(document.createTextNode(text.slice(cursor, start)));
        const span = document.createElement('mark');
        span.textContent = text.slice(start, end);
        container.append(span);
        cursor = end;
    }
    container.append(document.createTextNode(text.slice(cursor)));
    return ranges.length === new Set(quotes).size;
}

async function openSource(index, reference, trigger) {
    const chunk = reference.chunks[index];
    if (!chunk?.document_external_id) return;
    const current = ++sourceRequest;
    sourceTrigger = trigger;
    $('source').hidden = true;
    showError(null);
    try {
        const path = '/api/demo/source?' + new URLSearchParams({docId: chunk.document_external_id});
        const {body} = await api(path);
        if (current !== sourceRequest) return;
        const quotes = reference.spans.filter(span => span.citationIndex === index).map(span => span.text);
        $('source-title').textContent = body.title;
        $('source-meta').textContent = body.section + ' · 手册 ' + body.version;
        if (!highlightOriginal(body.text, quotes)) {
            $('source-meta').textContent += ' · 部分检索片段未能定位到原文';
        }
        $('source').hidden = false;
        $('source').focus({preventScroll: true});
        $('source').scrollIntoView({block: 'nearest'});
    } catch (error) {
        if (current === sourceRequest) showError(error);
    }
}

function citationButton(index, reference) {
    const control = button('[' + (index + 1) + ']', event =>
        openSource(index, reference, event.currentTarget), 'citation');
    control.setAttribute('aria-label', '查看引用 ' + (index + 1) + ' 原文');
    control.setAttribute('aria-controls', 'source');
    return control;
}

function showAnswer(body) {
    const {answer, answerStatus, reference} = body;
    const refused = answerStatus === 'insufficient_evidence';
    $('answer-status').textContent = refused ? '证据不足' : '已找到依据';
    $('answer-status').classList.toggle('refusal', refused);
    $('answer').classList.add('has-result');
    $('answer').replaceChildren();
    for (const line of answer.split('\n')) {
        const paragraph = document.createElement('p');
        let cursor = 0;
        for (const match of line.matchAll(/\[ID:(\d+)\]/g)) {
            const index = Number(match[1]);
            if (!reference.chunks[index]) continue;
            paragraph.append(document.createTextNode(line.slice(cursor, match.index)));
            paragraph.append(citationButton(index, reference));
            cursor = match.index + match[0].length;
        }
        paragraph.append(document.createTextNode(line.slice(cursor)));
        $('answer').append(paragraph);
    }
    $('sources').replaceChildren(...reference.chunks.map((chunk, index) =>
        button('[' + (index + 1) + '] ' + chunk.document_name,
            event => openSource(index, reference, event.currentTarget))));
    $('sources-region').hidden = reference.chunks.length === 0;
}

function showTrace(body, requestId, elapsedMs) {
    const evaluation = body.evaluation;
    const facts = body.retrievalTrace;
    const lexical = facts.find(stage => stage.stage === 'lexical_independent')?.returned ?? 0;
    const dense = facts.find(stage => stage.stage === 'dense_independent')?.returned ?? 0;
    const total = facts.find(stage => stage.stage === 'independent_rank')?.candidates ?? 0;
    $('trace-summary').textContent = 'BM25 ' + lexical + ' 条 · 向量 ' + dense + ' 条 · RRF ' + total
        + ' 条 · 检索 ' + Math.round(evaluation.retrievalMs) + ' ms · 请求 ' + Math.round(elapsedMs) + ' ms';
    $('candidates').replaceChildren(...evaluation.retrievedChunks.map(chunk => {
        const item = document.createElement('li');
        item.textContent = chunk.document_name + ' · ' + chunk.content;
        return item;
    }));
    $('trace-json').textContent = JSON.stringify({
        requestId, model: body.model, retrieval: facts, selection: {
            reason: evaluation.modelBoundary.selectionReason,
            passages: evaluation.modelBoundary.passageCount ?? 0,
            durationMs: evaluation.modelBoundary.modelMs ?? 0
        }
    }, null, 2);
    $('trace').hidden = false;
}

$('question-form').addEventListener('submit', event => {
    event.preventDefault();
    const question = $('question').value.trim();
    if (!ready() || !question) return;
    const version = $('version').value;
    perform(async () => {
        hideSource();
        $('answer-status').textContent = '';
        $('answer').classList.remove('has-result');
        $('answer').textContent = '正在检索并校验证据…';
        $('sources-region').hidden = true;
        $('trace').hidden = true;
        const started = performance.now();
        try {
            const {body, requestId} = await api('/api/demo/chat', {method: 'POST', payload: {question, version}});
            showAnswer(body);
            showTrace(body, requestId, performance.now() - started);
        } catch (error) {
            $('answer').textContent = '这次查询未完成。';
            throw error;
        }
    });
});

$('seed').addEventListener('click', () => perform(async () => {
    $('catalog-status').textContent = '正在建立示例索引…';
    try {
        showCatalog((await api('/api/demo', {method: 'POST'})).body);
    } catch (error) {
        $('catalog-status').textContent = '示例索引未就绪';
        throw error;
    }
}));
$('connect').addEventListener('click', () => perform(connect));
$('close-source').addEventListener('click', () => hideSource(true));
perform(connect);
