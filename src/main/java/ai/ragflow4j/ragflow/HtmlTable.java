// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0. See http://www.apache.org/licenses/LICENSE-2.0
// Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
package ai.ragflow4j.ragflow;

import java.util.*;

import org.jsoup.nodes.*;

/**
 * 对照 tableutil 的 ParseTableHTML/buildGrid/RenderTableHTML，处理跨行跨列并重建分块可用的表格。
 */
final class HtmlTable {
    private record Input(String text, int columns, int rows) {
    }

    private record Origin(int row, int column) {
    }

    private static final class Cell {
        String text;
        int columns, rows;
        boolean covered;

        Cell(String text, int columns, int rows) {
            this.text = text;
            this.columns = columns;
            this.rows = rows;
        }

        static Cell covered() {
            Cell cell = new Cell("", 0, 0);
            cell.covered = true;
            return cell;
        }
    }

    private final List<List<Input>> rows = new ArrayList<>();
    private final List<Boolean> headers = new ArrayList<>();
    private final List<List<Input>> footer = new ArrayList<>();
    private final List<Boolean> footerHeaders = new ArrayList<>();
    private String caption = "";

    static String render(Element table) {
        HtmlTable parsed = new HtmlTable();
        parsed.collect(table, false);
        parsed.rows.addAll(parsed.footer);
        parsed.headers.addAll(parsed.footerHeaders);
        int headerCount = 0;
        while (headerCount < parsed.headers.size() && parsed.headers.get(headerCount)) headerCount++;
        if (!parsed.rows.isEmpty() && headerCount == 0) headerCount = 1;
        List<List<Cell>> grid = grid(parsed.rows);
        StringBuilder result = new StringBuilder("<table>");
        if (!parsed.caption.isEmpty())
            result.append("<caption>").append(escape(parsed.caption)).append("</caption>");
        result.append('\n');
        for (int r = 0; r < grid.size(); r++) {
            result.append("<tr>");
            String tag = r < headerCount ? "th" : "td";
            for (Cell cell : grid.get(r)) {
                if (cell.covered) continue;
                result.append('<').append(tag);
                if (cell.columns > 1) result.append(" colspan=\"").append(cell.columns).append('"');
                if (cell.rows > 1) result.append(" rowspan=\"").append(cell.rows).append('"');
                result.append('>').append(escape(cell.text)).append("</").append(tag).append('>');
            }
            result.append("</tr>\n");
        }
        return result.append("</table>\n").toString();
    }

    private void collect(Node node, boolean inFooter) {
        if (node instanceof Element element) {
            String tag = element.normalName();
            if (inert(tag)) return;
            if (tag.equals("caption")) {
                caption = cellText(element);
                return;
            }
            if (tag.equals("tfoot")) {
                for (Node child : element.childNodes()) collect(child, true);
                return;
            }
            if (tag.equals("tr")) {
                List<Input> cells = new ArrayList<>();
                boolean header = false;
                for (Element child : element.children()) {
                    if (!Set.of("td", "th").contains(child.normalName())) continue;
                    header |= child.normalName().equals("th");
                    cells.add(
                            new Input(
                                    cellText(child), span(child, "colspan", 1000), span(child, "rowspan", 65534)));
                }
                (inFooter ? footer : rows).add(cells);
                (inFooter ? footerHeaders : headers).add(header);
                return;
            }
        }
        for (Node child : node.childNodes()) collect(child, inFooter);
    }

    private static List<List<Cell>> grid(List<List<Input>> source) {
        List<TreeMap<Integer, Origin>> covered = new ArrayList<>();
        for (int i = 0; i < source.size(); i++) covered.add(new TreeMap<>());
        int width = 0;
        for (int r = 0; r < source.size(); r++) {
            int column = 0, input = 0;
            while (true) {
                if (covered.get(r).containsKey(column)) column++;
                else if (input < source.get(r).size()) {
                    Input cell = source.get(r).get(input++);
                    column += cell.columns();
                    for (int dr = 1; dr < cell.rows() && r + dr < source.size(); dr++) {
                        for (int dc = 0; dc < cell.columns(); dc++)
                            covered.get(r + dr).put(column - 1 - dc, new Origin(r, column - cell.columns()));
                    }
                } else {
                    Integer next = covered.get(r).higherKey(column);
                    if (next == null) break;
                    column = next;
                }
                width = Math.max(width, column);
            }
        }
        List<List<Cell>> grid = new ArrayList<>();
        if (width == 0) return grid;
        for (int r = 0; r < source.size(); r++) {
            List<Cell> row = new ArrayList<>();
            int column = 0, input = 0;
            while (column < width) {
                if (covered.get(r).containsKey(column)) {
                    row.add(Cell.covered());
                    column++;
                    continue;
                }
                if (input >= source.get(r).size()) {
                    column++;
                    continue;
                }
                Input cell = source.get(r).get(input++);
                row.add(new Cell(cell.text(), cell.columns(), cell.rows()));
                for (int dc = 1; dc < cell.columns(); dc++) row.add(Cell.covered());
                column += cell.columns();
            }
            grid.add(row);
        }
        for (int r = 0; r < grid.size(); r++) {
            for (int c = 0; c < grid.get(r).size(); c++) {
                Cell cell = grid.get(r).get(c);
                if (!cell.covered) {
                    cell.rows = Math.min(cell.rows, grid.size() - r);
                    cell.columns = Math.min(cell.columns, grid.get(r).size() - c);
                }
            }
        }
        while (!grid.isEmpty() && grid.getLast().isEmpty()) grid.removeLast();
        return grid;
    }

    private static int span(Element cell, String key, int max) {
        try {
            return Math.max(1, Math.min(max, Integer.parseInt(cell.attr(key).strip())));
        } catch (NumberFormatException ignored) {
            return 1;
        }
    }

    static boolean inert(String tag) {
        return Set.of("script", "style", "noscript", "template").contains(tag);
    }

    private static String cellText(Node node) {
        StringBuilder text = new StringBuilder();
        collectText(node, text);
        return text.toString().strip();
    }

    private static void collectText(Node node, StringBuilder text) {
        if (node instanceof TextNode leaf) text.append(leaf.getWholeText());
        if (node instanceof Element element) {
            if (inert(element.normalName())) return;
            if (element.normalName().equals("br")) text.append('\n');
        }
        for (Node child : node.childNodes()) collectText(child, text);
    }

    private static String escape(String text) {
        return text.replace("&", "&amp;")
                .replace("'", "&#39;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&#34;");
    }
}
