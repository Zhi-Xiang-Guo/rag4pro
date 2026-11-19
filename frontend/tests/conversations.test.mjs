import assert from 'node:assert/strict';
import test from 'node:test';
import {decodeHistory, recentHistory} from '../src/conversations.ts';

test('loads valid saved chats around damaged entries', () => {
  const history = decodeHistory(JSON.stringify({currentId: 'chat', sessions: [
    null,
    {id: 'broken', messages: null},
    {id: 'chat', title: '资料', datasetId: 'kb', docId: 'doc', version: 'v1', messages: [
      null,
      {id: 'q', role: 'user', content: '问题'},
      {id: 'broken', role: 'assistant', content: 42},
      {id: 'a', role: 'assistant', content: '结果 [ID:0]', model: 'model', reference: {
        chunks: [{document_name: '资料', content: '原文', source: {revision: 'abc', kind: 'page', unit: 3}}],
        spans: [null, {citationIndex: 0, start: 0, end: 2, text: '原文'}],
      }},
    ]},
  ]}));
  assert.equal(history.currentId, 'chat');
  assert.equal(history.sessions.length, 1);
  const session = history.sessions[0];
  assert.equal(session.datasetId, 'kb');
  assert.equal(session.docId, 'doc');
  assert.equal(session.version, 'v1');
  assert.deepEqual(session.messages.map(message => message.content), ['问题', '结果 [ID:0]']);
  assert.equal(session.messages[1].reference.chunks[0].source.unit, 3);
  assert.equal(session.messages[1].reference.spans.length, 1);
});

test('clears missing selected sessions and ignores unreadable saved data', () => {
  assert.deepEqual(decodeHistory('{'), {currentId: '', sessions: []});
  assert.deepEqual(decodeHistory(JSON.stringify({currentId: 'gone', sessions: []})), {currentId: '', sessions: []});
});

test('request context keeps complete recent turns and the new question', () => {
  const messages = [{role: 'assistant', content: 'orphan'}];
  for (let turn = 1; turn <= 7; turn += 1) {
    messages.push({role: 'user', content: 'q' + turn}, {role: 'assistant', content: 'a' + turn});
  }
  messages.push({role: 'user', content: 'unfinished'});
  const history = recentHistory(messages, 'next');
  assert.equal(history.length, 11);
  assert.deepEqual(history.slice(0, 2), [{role: 'user', content: 'q3'}, {role: 'assistant', content: 'a3'}]);
  assert.deepEqual(history.at(-1), {role: 'user', content: 'next'});
  assert.ok(history.every(message => !['orphan', 'unfinished'].includes(message.content)));
});
