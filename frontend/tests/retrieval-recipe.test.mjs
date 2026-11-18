import assert from 'node:assert/strict';
import test from 'node:test';
import { readRecipe } from '../src/retrievalRecipe.ts';

const example = {
  version: 1, datasetId: 'runbook', docId: 'refund', question: 'Why?', topK: 4, candidateK: 32,
  evidenceBudget: 1200, baseline: {strategy: 'standard', contextMode: 'leaf', useReranker: true},
  comparison: {strategy: 'rrf', contextMode: 'parent', useReranker: false},
};

test('round-trips valid settings and ignores unrelated keys', () => {
  const text = JSON.stringify({...example, token: 'extra', comparison: {...example.comparison, other: 'extra'}});
  assert.deepEqual(readRecipe(JSON.parse(text)), example);
});

test('rejects invalid scope, budgets, settings and versions', () => {
  assert.throws(() => readRecipe({...example, version: 2}), /版本/);
  assert.throws(() => readRecipe({...example, docId: 1}), /范围/);
  assert.throws(() => readRecipe({...example, topK: 40}), /候选池/);
  assert.throws(() => readRecipe({...example, topK: 2.5}), /Top K/);
  assert.throws(() => readRecipe({...example, evidenceBudget: 9000}), /证据预算/);
  assert.throws(() => readRecipe({...example, comparison: {...example.comparison, useReranker: true}}), /精排/);
});
