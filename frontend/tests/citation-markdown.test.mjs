import assert from 'node:assert/strict';
import test from 'node:test';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {citationMarkdown} from '../src/citationMarkdown.ts';

function render(text) {
  return renderToStaticMarkup(createElement(Markdown, {
    children: text, skipHtml: true, remarkPlugins: [remarkGfm, citationMarkdown(2)],
    components: {span: ({node, children}) => createElement('button', {
      'data-ref': node.properties['data-citation-index'],
    }, children)},
  }));
}

test('recognizes adjacent citations and leaves unknown markers visible', () => {
  assert.equal(render('**结论 [ID:0][ID:1]** [ID:2]'),
    '<p><strong>结论 <button data-ref="0">[1]</button><button data-ref="1">[2]</button></strong> [ID:2]</p>');
});

test('keeps code and existing links intact', () => {
  assert.equal(render('`[ID:0]` [[ID:1]](https://example.org) [link](#citation-0)'),
    '<p><code>[ID:0]</code> <a href="https://example.org">[ID:1]</a> <a href="#citation-0">link</a></p>');
  assert.match(render('```\n[ID:0]\n```'), /<pre><code>\[ID:0\]/);
});
