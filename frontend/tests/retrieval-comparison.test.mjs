import assert from 'node:assert/strict';
import test from 'node:test';
import { buildExperimentReport, compareChunks, describeStage } from '../src/retrievalComparison.ts';

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

test('exports budgets, ranks and counts without query text, bodies or raw traces', () => {
  const recipe = {
    version: 1, datasetId: 'd', docId: 'refund', question: 'private input',
    topK: 2, candidateK: 16, evidenceBudget: 600,
    baseline: {strategy: 'standard', contextMode: 'leaf', useReranker: false},
    comparison: {strategy: 'rrf', contextMode: 'leaf', useReranker: false},
  };
  const chunk = {id: 'a', source: {revision: 'r1'}, content: 'private body'};
  const baseline = {
    hits: [{}, chunk], evidence: [chunk], evidenceTokens: 111, retrievalMs: 31, total: 12,
    contextMode: 'leaf', trace: [{stage: 'initial', returned: 12, request: {question: 'private trace'}}],
  };
  const comparison = {...baseline, hits: [chunk], trace: []};
  const report = buildExperimentReport(recipe, '2026-11-27T00:00:00Z', baseline, comparison);

  assert.deepEqual(report.baseline.hits, [{id: 'a', rank: 2, sourceRevision: 'r1'}]);
  assert.deepEqual(report.overlap.hits, {shared: 1, baselineOnly: 0, comparisonOnly: 0});
  assert.equal(report.budgets.candidateK, 16);
  assert.equal(report.comparison.settings.strategy, 'rrf');
  assert.doesNotMatch(JSON.stringify(report), /private/);
});
