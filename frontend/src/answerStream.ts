import type {Answer} from './answers';

type Event = {type: string; data: string};

async function* parseEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<Event> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let type = '';
  let data: string[] = [];
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      let lineEnd = buffer.search(/[\r\n]/);
      while (lineEnd >= 0 && (done || buffer[lineEnd] !== '\r' || lineEnd < buffer.length - 1)) {
        const line = buffer.slice(0, lineEnd);
        const separatorLength = buffer.slice(lineEnd, lineEnd + 2) === '\r\n' ? 2 : 1;
        buffer = buffer.slice(lineEnd + separatorLength);
        if (line === '') {
          if (data.length) yield {type, data: data.join('\n')};
          type = '';
          data = [];
        } else if (!line.startsWith(':')) {
          const split = line.indexOf(':');
          const field = split < 0 ? line : line.slice(0, split);
          const value = split < 0 ? '' : line.slice(split + 1).replace(/^ /, '');
          if (field === 'event') type = value;
          if (field === 'data') data.push(value);
        }
        lineEnd = buffer.search(/[\r\n]/);
      }
      if (done) return;
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export async function readAnswerStream(stream: ReadableStream<Uint8Array>, onDelta: (text: string) => void): Promise<Answer> {
  for await (const event of parseEvents(stream)) {
    if (!['delta', 'result', 'error'].includes(event.type)) continue;
    let payload;
    try { payload = JSON.parse(event.data); }
    catch { throw new Error('流式回答返回了无效的数据。'); }
    if (event.type === 'error') {
      throw new Error(typeof payload?.error === 'string' && payload.error || '流式回答失败。');
    }
    if (event.type === 'delta' && typeof payload?.text === 'string') {
      onDelta(payload.text);
    } else if (event.type === 'result' && typeof payload?.answer === 'string') {
      return payload as Answer;
    } else {
      throw new Error('流式回答返回了无效的数据。');
    }
  }
  throw new Error('流式连接在返回正式答案前断开。');
}
