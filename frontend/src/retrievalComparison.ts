import type { Evidence } from './answers';
import type { RetrievalRecipe } from './retrievalRecipe';

export function compareChunks(baseline: Evidence[], comparison: Evidence[]) {
  const rows = new Map<string, {chunk: Evidence; baseline?: number; comparison?: number}>();
  for (const [arm, chunks] of [['baseline', baseline], ['comparison', comparison]] as const) {
    chunks.forEach((chunk, index) => {
      if (!chunk.id) return;
      const row = rows.get(chunk.id) ?? {chunk};
      row[arm] ??= index + 1;
      rows.set(chunk.id, row);
    });
  }
  const changes = Array.from(rows, ([id, row]) => ({id, ...row}));
  const shared = changes.filter(row => row.baseline !== undefined && row.comparison !== undefined).length;
  return {
    shared,
    baselineOnly: changes.filter(row => row.comparison === undefined).length,
    comparisonOnly: changes.filter(row => row.baseline === undefined).length,
    changes,
  };
}

const stageLabels: Record<string, string> = {
  fulltext: '纯词法召回', initial: '首次召回', relaxed: '放宽匹配', dense_fallback: '向量回退',
  lexical_independent: 'BM25 召回', dense_independent: '向量召回', independent_rank: '独立召回排序',
  rerank_model: '模型精排', vector_only_scores: '使用向量引擎分数',
  local_rerank_fallback: '回退到本地评分', knn_scores: '候选向量评分', pagination: '阈值过滤与分页',
};

export function describeStage(step: Record<string, unknown>) {
  const stage = typeof step.stage === 'string' ? step.stage : '';
  const values: [string, unknown][] = [
    ['引擎匹配', step.total], ['返回', step.returned], ['候选', step.candidates ?? step.candidateCount],
    ['过阈值', step.validCount], ['阈值', step.threshold], ['RRF 常数', step.rrfK],
  ];
  const details = values.flatMap(([label, value]) =>
    typeof value === 'number' && Number.isFinite(value) ? [label + ' ' + value] : []);
  if (typeof step.durationMs === 'number' && Number.isFinite(step.durationMs)) {
    details.push(step.durationMs.toFixed(1) + ' ms');
  }
  if (step.thresholdApplied === false) details.push('未使用相似度阈值');
  if (step.documentDiversity === true) details.push('优先覆盖不同文档');
  return {label: stageLabels[stage] ?? (stage || '检索'), details};
}

export type RetrievalResult = {
  hits: Evidence[];
  evidence: Evidence[];
  evidenceTokens: number;
  retrievalMs: number;
  contextMode: string;
  total: number;
  trace: Record<string, unknown>[];
};

export function buildExperimentReport(recipe: RetrievalRecipe, startedAt: string,
  baseline: RetrievalResult, comparison: RetrievalResult) {
  const chunks = (list: Evidence[]) => list.flatMap((chunk, index) => chunk.id ? [{
    id: chunk.id, rank: index + 1, sourceRevision: chunk.source?.revision,
  }] : []);
  const summarize = (result: RetrievalResult) => ({
    candidateTotal: result.total,
    retrievalMs: result.retrievalMs,
    evidenceTokens: result.evidenceTokens,
    hits: chunks(result.hits),
    evidence: chunks(result.evidence),
    stages: result.trace.map(describeStage),
  });
  const overlap = (left: Evidence[], right: Evidence[]) => {
    const {shared, baselineOnly, comparisonOnly} = compareChunks(left, right);
    return {shared, baselineOnly, comparisonOnly};
  };
  return {
    version: 1,
    startedAt,
    scope: {datasetId: recipe.datasetId, docId: recipe.docId},
    budgets: {topK: recipe.topK, candidateK: recipe.candidateK, evidenceBudget: recipe.evidenceBudget},
    baseline: {settings: recipe.baseline, ...summarize(baseline)},
    comparison: {settings: recipe.comparison, ...summarize(comparison)},
    overlap: {hits: overlap(baseline.hits, comparison.hits), evidence: overlap(baseline.evidence, comparison.evidence)},
  };
}
