import assert from 'node:assert/strict';
import test from 'node:test';
import {readAnswerStream} from '../src/answerStream.ts';

const encoder = new TextEncoder();

function stream(source, step = source.length) {
  const bytes = encoder.encode(source);
  return new ReadableStream({
    start(controller) {
      for (let index = 0; index < bytes.length; index += step) {
        controller.enqueue(bytes.slice(index, index + step));
      }
      controller.close();
    },
  });
}

test('decodes mixed event separators, split UTF-8 and multiline data', async () => {
  const output = [];
  const response = 'event: start\r\ndata: ignored\r\n\r\n: heartbeat\r\r'
    + 'event: delta\rdata: {"text":"内容"}\r\r'
    + 'event: delta\r\ndata: {"text":\r\ndata: "🏷️"}\r\n\r\n'
    + 'event: result\ndata: {"answer":"正式答案","model":"test"}\n\n';
  const result = await readAnswerStream(stream(response, 1), value => output.push(value));
  assert.deepEqual(output, ['内容', '🏷️']);
  assert.deepEqual(result, {answer: '正式答案', model: 'test'});
});

test('reports streamed failures, invalid payloads and disconnects separately', async () => {
  const noop = () => {};
  await assert.rejects(readAnswerStream(stream('event: error\ndata: {"error":"模型不可用"}\n\n'), noop), /模型不可用/);
  await assert.rejects(readAnswerStream(stream('event: delta\ndata: {"text":42}\n\n'), noop), /无效的数据/);
  await assert.rejects(readAnswerStream(stream('event: result\ndata: bad\n\n'), noop), /无效的数据/);
  await assert.rejects(readAnswerStream(stream('event: delta\ndata: {"text":"部分"}\n\n'), noop), /正式答案前断开/);
});

test('releases the body and stops reading when a final answer arrives', async () => {
  let cancelled = false;
  const body = new ReadableStream({
    start(controller) { controller.enqueue(encoder.encode('event: result\ndata: {"answer":"done"}\n\n')); },
    cancel() { cancelled = true; },
  });
  assert.deepEqual(await readAnswerStream(body, () => {}), {answer: 'done'});
  assert.equal(cancelled, true);
  assert.equal(body.locked, false);
});
