import { Fragment } from 'react';
import type { TextRange } from '../citationRanges';

export function HighlightedText({text, ranges}: {text: string; ranges: readonly TextRange[]}) {
  if (!ranges.length) return <>{text}</>;
  return <>{ranges.map(({start, end}, index) =>
    <Fragment key={start}>{text.slice(ranges[index - 1]?.end ?? 0, start)}<mark>{text.slice(start, end)}</mark></Fragment>
  )}{text.slice(ranges[ranges.length - 1].end)}</>;
}
