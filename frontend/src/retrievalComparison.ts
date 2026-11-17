import type { Evidence } from './answers';

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
