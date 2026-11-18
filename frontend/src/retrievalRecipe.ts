const strategyNames = ['standard', 'lexical', 'dense', 'rrf', 'rrf_diverse'] as const;
const contextModes = ['leaf', 'parent', 'section'] as const;

export type RetrievalArm = {
  strategy: typeof strategyNames[number];
  contextMode: typeof contextModes[number];
  useReranker: boolean;
};

export type RetrievalRecipe = {
  version: 1;
  datasetId: string;
  docId?: string;
  question: string;
  topK: number;
  candidateK: number;
  evidenceBudget: number;
  baseline: RetrievalArm;
  comparison: RetrievalArm;
};

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

function boundedInteger(value: unknown, min: number, max: number, label: string) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
    throw new Error(label + '须在 ' + min + '–' + max + ' 之间。');
  }
  return value;
}

function readArm(value: unknown, label: string): RetrievalArm {
  if (!isObject(value)) throw new Error(label + '缺少检索设置。');
  const strategy = strategyNames.find(name => name === value.strategy);
  const contextMode = contextModes.find(mode => mode === value.contextMode);
  if (!strategy || !contextMode || typeof value.useReranker !== 'boolean') {
    throw new Error(label + '检索设置无效。');
  }
  if (value.useReranker && strategy !== 'standard') {
    throw new Error(label + '所选召回方式暂不支持精排。');
  }
  return {strategy, contextMode, useReranker: value.useReranker};
}

export function readRecipe(value: unknown): RetrievalRecipe {
  if (!isObject(value) || value.version !== 1) throw new Error('实验配置版本无效。');
  const {datasetId, docId, question} = value;
  if (typeof datasetId !== 'string' || !datasetId.trim() || datasetId.length > 1024 ||
      (docId !== undefined && (typeof docId !== 'string' || !docId.trim() || docId.length > 1024))) {
    throw new Error('配置中的知识库或文档范围无效。');
  }
  if (typeof question !== 'string' || !question.trim() || question.length > 2000) {
    throw new Error('请输入 1–2000 字符的检索问题。');
  }
  const topK = boundedInteger(value.topK, 1, 50, '最终 Top K');
  const candidateK = boundedInteger(value.candidateK, Math.max(2, topK), 256, '候选池');
  const evidenceBudget = boundedInteger(value.evidenceBudget, 1, 8192, '证据预算');
  return {
    version: 1, datasetId, docId, question: question.trim(), topK, candidateK, evidenceBudget,
    baseline: readArm(value.baseline, '基线'),
    comparison: readArm(value.comparison, '对照'),
  };
}

export function downloadExperiment(filename: string, payload: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
