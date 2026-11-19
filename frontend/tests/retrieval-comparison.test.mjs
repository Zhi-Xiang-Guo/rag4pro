import assert from 'node:assert/strict';
import test from 'node:test';
import { compareChunks, describeStage } from '../src/retrievalComparison.ts';

test('compares stable chunk identity and preserves the first rank', () => {
  const baseline = [{id: 'a'}, {id: 'a'}, {id: 'b'}, {content: 'unidentified'}];
  const comparison = [{id: 'c'}, {id: 'b'}, {id: 'a'}];
  const result = compareChunks(baseline, comparison);

  assert.deepEqual([result.shared, result.baselineOnly, result.comparisonOnly], [2, 0, 1]);
  assert.deepEqual(result.changes.map(row => [row.id, row.baseline, row.comparison]), [
    ['a', 1, 3], ['b', 3, 2], ['c', undefined, 1],
  ]);
});

test('counts exclusive chunks when the other side has no hits', () => {
  assert.deepEqual(compareChunks([{id: 'a'}], []), {
    shared: 0,
    baselineOnly: 1,
    comparisonOnly: 0,
    changes: [{id: 'a', chunk: {id: 'a'}, baseline: 1}],
  });
});

test('selects numeric trace diagnostics and excludes raw engine requests', () => {
  const first = describeStage({stage: 'initial', total: 0, returned: 0, durationMs: 2.04, request: {vector: [1, 2]}});
  assert.deepEqual(first, {label: '首次召回', details: ['引擎匹配 0', '返回 0', '2.0 ms']});
  const final = describeStage({stage: 'independent_rank', candidates: 8, rrfK: 60,
    documentDiversity: true, thresholdApplied: false});
  assert.deepEqual(final.details, ['候选 8', 'RRF 常数 60', '未使用相似度阈值', '优先覆盖不同文档']);
  assert.deepEqual(describeStage({stage: 'future', candidates: NaN, total: {value: 200}}),
    {label: 'future', details: []});
});
