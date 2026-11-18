export type TextRange = {start: number; end: number};
type Quote = TextRange & {text: string};

function uniqueOffset(text: string, quote: string) {
  const first = text.indexOf(quote);
  return first >= 0 && text.indexOf(quote, first + 1) < 0 ? first : -1;
}

export function quoteRanges(content: string, quotes: readonly Quote[]): TextRange[] {
  const valid = quotes.filter(({start, end, text}) => Number.isInteger(start) && Number.isInteger(end)
    && start >= 0 && end > start && end <= content.length && content.slice(start, end) === text);
  return mergeRanges(valid);
}

function mergeRanges(ranges: readonly TextRange[]) {
  const sorted = [...ranges].sort((first, second) => first.start - second.start);
  const merged: TextRange[] = [];
  for (const {start, end} of sorted) {
    const last = merged.at(-1);
    if (last && start <= last.end) last.end = Math.max(last.end, end);
    else merged.push({start, end});
  }
  return merged;
}

export function sourceRanges(source: string, content: string | undefined, offset: number | undefined,
  quotes: readonly Quote[]): TextRange[] {
  if (!content) return [];
  const ranges = quoteRanges(content, quotes);
  const selected = ranges.length ? ranges : [{start: 0, end: content.length}];
  const anchored = offset !== undefined && offset >= 0 && source.slice(offset, offset + content.length) === content;
  const contentOffset = anchored ? offset : uniqueOffset(source, content);
  if (contentOffset >= 0) {
    return selected.map(({start, end}) => ({start: contentOffset + start, end: contentOffset + end}));
  }
  return mergeRanges(selected.flatMap(({start, end}) => {
    const found = uniqueOffset(source, content.slice(start, end));
    return found < 0 ? [] : [{start: found, end: found + end - start}];
  }));
}
