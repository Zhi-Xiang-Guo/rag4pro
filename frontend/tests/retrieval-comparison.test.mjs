import assert from 'node:assert/strict';
import test from 'node:test';
import { compareChunks } from '../src/retrievalComparison.ts';

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
