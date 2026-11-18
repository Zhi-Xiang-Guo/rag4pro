import assert from 'node:assert/strict';
import test from 'node:test';
import {quoteRanges, sourceRanges} from '../src/citationRanges.ts';

test('validates exact UTF-16 ranges and keeps different cited sentences', () => {
  const content = '🧪甲句。中间。乙句。';
  const first = {start: 2, end: 5, text: '甲句。'};
  const second = {start: 8, end: 11, text: '乙句。'};
  assert.deepEqual(quoteRanges(content, [
    second, {start: -1, end: 2, text: '🧪'}, first, {start: 3, end: 5, text: '句。'},
    {start: 8, end: 11, text: '甲句。'},
  ]), [{start: 2, end: 5}, {start: 8, end: 11}]);
});

test('maps repeated quotes using the cited chunk offset or a unique whole chunk', () => {
  const content = '甲乙甲';
  const quote = {start: 2, end: 3, text: '甲'};
  assert.deepEqual(sourceRanges('甲乙甲为甲乙甲', content, 4, [quote]), [{start: 6, end: 7}]);
  assert.deepEqual(sourceRanges('甲。甲乙甲。', content, 1, [quote]), [{start: 4, end: 5}]);
  assert.deepEqual(sourceRanges('甲乙甲为甲乙甲', content, 1, [quote]), []);
});

test('can highlight independent unique quotes when formatting differs', () => {
  const content = '段落 引文 示例';
  assert.deepEqual(sourceRanges('段落\n引文\n示例', content, 0, [
    {start: 0, end: 2, text: '段落'},
    {start: 6, end: 8, text: '示例'},
  ]), [{start: 0, end: 2}, {start: 6, end: 8}]);
});
