// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations.
package ai.ragflow4j.ragflow;

import java.util.*;

import org.commonmark.ext.gfm.tables.TableBlock;
import org.commonmark.ext.gfm.tables.TablesExtension;
import org.commonmark.node.*;
import org.commonmark.parser.IncludeSourceSpans;
import org.commonmark.parser.Parser;
import org.commonmark.renderer.html.HtmlRenderer;

/**
 * 将 Markdown 语法树遍历为标题、正文、表格等解析单元，交给 GeneralChunker 进一步合并。
 *
 * <p>对照 internal/parser/parser/markdown_parser.go；无法完整处理的内容通过 warnings 返回。
 */
public final class MarkdownParser {
    public record Result(List<Map<String, Object>> units, List<String> warnings) {
    }

    private final List<org.commonmark.Extension> extensions = List.of(TablesExtension.create());
    private final Parser parser =
            Parser.builder().extensions(extensions).includeSourceSpans(IncludeSourceSpans.BLOCKS).build();
    private final HtmlRenderer renderer = HtmlRenderer.builder().extensions(extensions).build();

    public Result parse(String markdown) {
        if (markdown.startsWith("\ufeff")) markdown = markdown.substring(1);
        String normalized = GeneralChunker.normalize(markdown) + "\n\n";
        Node document = parser.parse(balanceTables(normalized));
        List<Map<String, Object>> units = new ArrayList<>();
        Set<String> warnings = new LinkedHashSet<>();
        walkBlocks(document, units, warnings);
        if (units.isEmpty()) units.add(GeneralChunker.unit("", ""));
        return new Result(List.copyOf(units), List.copyOf(warnings));
    }

    private void walkBlocks(Node parent, List<Map<String, Object>> units, Set<String> warnings) {
        for (Node node = parent.getFirstChild(); node != null; node = node.getNext()) {
            if (node instanceof TableBlock
                    || node instanceof HtmlBlock html && isTableHtml(html.getLiteral())) {
                units.addAll(HtmlTextParser.parse(renderer.render(node), warnings));
            } else if (node instanceof BulletList || node instanceof OrderedList) {
                int index = 0;
                for (Node item = node.getFirstChild(); item != null; item = item.getNext()) {
                    if (containsTable(item)) {
                        int start = units.size();
                        walkBlocks(item, units, warnings);
                        for (int i = start; i < units.size(); i++) {
                            Map<String, Object> unit = units.get(i);
                            if ("text".equals(unit.get("doc_type_kwd"))) unit.putIfAbsent("layout", "list");
                        }
                        if (units.size() > start && "text".equals(units.get(start).get("doc_type_kwd"))) {
                            units.get(start).put("text", marker(node, index) + units.get(start).get("text"));
                        }
                    } else {
                        String text = leaf(item).strip();
                        if (!text.isEmpty()) units.add(GeneralChunker.unit(marker(node, index) + text, "list"));
                    }
                    index++;
                }
            } else if (node instanceof BlockQuote && containsTable(node)) {
                walkBlocks(node, units, warnings);
            } else if (node instanceof Paragraph && containsTable(node)) {
                units.addAll(HtmlTextParser.parse(renderer.render(node), warnings));
            } else {
                String text = leaf(node).strip();
                String layout = "";
                if (node instanceof Heading heading) {
                    text = "#".repeat(heading.getLevel()) + (text.isEmpty() ? "" : " " + text);
                    layout = "heading";
                } else if (node instanceof FencedCodeBlock || node instanceof IndentedCodeBlock)
                    layout = "code";
                if (!text.isEmpty()) units.add(GeneralChunker.unit(text, layout));
            }
            node.accept(
                    new AbstractVisitor() {
                        @Override
                        public void visit(Image image) {
                            warnings.add(
                                    "Markdown images retain textual references/alt text; image downloading, decoding"
                                            + " and OCR are unavailable.");
                        }
                    });
        }
    }

    private static boolean isTableHtml(String text) {
        return text.strip().toLowerCase(Locale.ROOT).matches("(?s)^<table[> \t\r\n].*");
    }

    private static boolean containsTable(Node node) {
        if (node instanceof TableBlock
                || node instanceof HtmlInline html && isTableHtml(html.getLiteral())
                || node instanceof HtmlBlock block && isTableHtml(block.getLiteral())) return true;
        for (Node child = node.getFirstChild(); child != null; child = child.getNext())
            if (containsTable(child)) return true;
        return false;
    }

    /**
     * 对应 markdown_parser.go 的 balanceTableColumns 和 markdownTableCells，补齐表格列。
     */
    private String balanceTables(String markdown) {
        Set<Integer> excluded = new HashSet<>();
        Node original = parser.parse(markdown);
        original.accept(
                new AbstractVisitor() {
                    private void exclude(Node node) {
                        for (SourceSpan span : node.getSourceSpans()) excluded.add(span.getLineIndex());
                    }

                    @Override
                    public void visit(FencedCodeBlock node) {
                        exclude(node);
                    }

                    @Override
                    public void visit(IndentedCodeBlock node) {
                        exclude(node);
                    }

                    @Override
                    public void visit(HtmlBlock node) {
                        exclude(node);
                    }
                });
        List<String> lines = new ArrayList<>(Arrays.asList(markdown.split("\n", -1)));
        for (int i = 0; i + 1 < lines.size(); i++) {
            if (excluded.contains(i) || excluded.contains(i + 1)) continue;
            int header = cells(lines.get(i)).size();
            if (header < 2 || !separator(lines.get(i + 1))) continue;
            int widest = header;
            int end = i + 2;
            while (end < lines.size() && !excluded.contains(end) && cells(lines.get(end)).size() >= 2) {
                widest = Math.max(widest, cells(lines.get(end)).size());
                end++;
            }
            if (widest > header) {
                lines.set(i, pad(lines.get(i), widest - header, false));
                lines.set(i + 1, pad(lines.get(i + 1), widest - header, true));
            }
            i = end - 1;
        }
        return String.join("\n", lines);
    }

    private static String pad(String line, int count, boolean separator) {
        line = line.replaceFirst("[ \t]+$", "");
        if (!line.endsWith("|")) line += "|";
        return line + (separator ? "---|" : " |").repeat(count);
    }

    private static boolean separator(String line) {
        List<String> cells = cells(line);
        if (cells.size() < 2) return false;
        for (String cell : cells) {
            cell = cell.strip().replace(" ", "");
            if (cell.startsWith(":")) cell = cell.substring(1);
            if (cell.endsWith(":")) cell = cell.substring(0, cell.length() - 1);
            if (cell.isEmpty() || !cell.matches("-+")) return false;
        }
        return true;
    }

    private static List<String> cells(String line) {
        line = line.strip();
        List<String> cells = new ArrayList<>();
        int start = 0;
        boolean atStart = true;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == ' ') continue;
            if (c == '\\') {
                i++;
                atStart = false;
                continue;
            }
            if (c == '`') {
                if (!atStart) continue;
                atStart = false;
                int end = i + 1;
                while (end < line.length() && line.charAt(end) == '`') end++;
                String marker = line.substring(i, end);
                int close = line.indexOf(marker, end);
                i = close >= 0 ? close + marker.length() - 1 : end - 1;
            } else if (c == '|') {
                if (i > 0 || start > 0) cells.add(line.substring(start, i));
                start = i + 1;
                atStart = true;
            } else atStart = false;
        }
        if (start == 0) return List.of();
        if (start < line.length()) cells.add(line.substring(start));
        return cells;
    }

    private static String marker(Node list, int index) {
        if (list instanceof OrderedList ordered)
            return (ordered.getStartNumber() + index) + String.valueOf(ordered.getDelimiter()) + " ";
        return ((BulletList) list).getBulletMarker() + " ";
    }

    private static String leaf(Node node) {
        if (node instanceof Text text) return text.getLiteral();
        if (node instanceof Code code) return code.getLiteral();
        if (node instanceof FencedCodeBlock code) return code.getLiteral();
        if (node instanceof IndentedCodeBlock code) return code.getLiteral();
        if (node instanceof HtmlBlock html) return html.getLiteral();
        if (node instanceof HtmlInline html) return html.getLiteral();
        if (node instanceof SoftLineBreak || node instanceof HardLineBreak) return "\n";
        StringBuilder value = new StringBuilder();
        boolean separate =
                node instanceof ListItem
                        || node instanceof BlockQuote
                        || node instanceof BulletList
                        || node instanceof OrderedList;
        for (Node child = node.getFirstChild(); child != null; child = child.getNext()) {
            if (separate && child != node.getFirstChild()) value.append('\n');
            value.append(leaf(child));
        }
        if (node instanceof Image image) {
            String destination = image.getDestination();
            String lower = destination.strip().toLowerCase(Locale.ROOT);
            if (!lower.startsWith("data:")
                    && !lower.startsWith("http://")
                    && !lower.startsWith("https://")) {
                return "![" + value + "](" + destination + ")";
            }
        }
        return value.toString();
    }
}
