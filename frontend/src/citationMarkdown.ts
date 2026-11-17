type MarkdownNode = {
  type: string;
  value?: string;
  children?: MarkdownNode[];
  data?: {hName?: string; hProperties?: Record<string, string>};
};

function expand(node: MarkdownNode, citationCount: number): MarkdownNode[] {
  if (node.type === 'link' || node.type === 'linkReference') return [node];
  if (node.children) {
    node.children = node.children.flatMap(child => expand(child, citationCount));
  }
  if (node.type !== 'text' || !node.value) return [node];

  const text = node.value;
  const parts: MarkdownNode[] = [];
  let offset = 0;
  for (const match of text.matchAll(/\[ID:(\d+)\]/g)) {
    const index = Number(match[1]);
    if (index >= citationCount) continue;
    if (offset < match.index) parts.push({type: 'text', value: text.slice(offset, match.index)});
    parts.push({
      type: 'strong',
      data: {hName: 'span', hProperties: {'data-citation-index': String(index)}},
      children: [{type: 'text', value: '[' + (index + 1) + ']'}],
    });
    offset = match.index + match[0].length;
  }
  if (!parts.length) return [node];
  if (offset < text.length) parts.push({type: 'text', value: text.slice(offset)});
  return parts;
}

export function citationMarkdown(citationCount: number) {
  return () => (tree: MarkdownNode) => { expand(tree, citationCount); };
}
