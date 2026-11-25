// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0. See http://www.apache.org/licenses/LICENSE-2.0
// Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
package ai.ragflow4j.ragflow;

import java.util.*;

import org.jsoup.Jsoup;
import org.jsoup.nodes.*;

/**
 * 对照 html_parser.go 的 leafWriter 遍历 HTML DOM，提取文本并保留表格结构，供 Markdown 解析复用。
 */
final class HtmlTextParser {
    private static final Set<String> BLOCKS =
            Set.of(
                    ("h1 h2 h3 h4 h5 h6 p li pre tr blockquote div section article header footer main nav"
                            + " aside address hgroup search center ul ol menu dl dt dd figure figcaption details"
                            + " summary dialog form fieldset legend hr")
                            .split(" "));
    private static final Set<String> INLINE =
            Set.of(
                    ("a abbr b bdi bdo big br cite code data del dfn em font i img ins kbd label mark q s"
                            + " samp small span strike strong sub sup time tt u var wbr")
                            .split(" "));

    static List<Map<String, Object>> parse(String markup, Set<String> warnings) {
        List<Map<String, Object>> result = new ArrayList<>();
        Node body = Jsoup.parse(markup).body();
        Writer run = new Writer();
        for (Node node : body.childNodes()) {
            if (node instanceof Comment || node instanceof Element e && HtmlTable.inert(e.normalName()))
                continue;
            if (node instanceof TextNode
                    || node instanceof Element e && INLINE.contains(e.normalName())) {
                walk(node, run, result, "text", true, warnings);
                continue;
            }
            run.flush(result, "text", true);
            if (!(node instanceof Element element)) continue;
            String tag = element.normalName();
            if (tag.equals("table")) emitTable(element, result, warnings);
            else {
                Writer writer = new Writer();
                boolean trim = !tag.equals("pre") && !tag.equals("textarea");
                walk(element, writer, result, layout(tag), trim, warnings);
                writer.flush(result, layout(tag), trim);
            }
        }
        run.flush(result, "text", true);
        return result;
    }

    private static void walk(
            Node node,
            Writer writer,
            List<Map<String, Object>> out,
            String layout,
            boolean trim,
            Set<String> warnings) {
        if (node instanceof TextNode leaf) {
            writer.text(leaf.getWholeText());
            return;
        }
        if (!(node instanceof Element element)) return;
        String tag = element.normalName();
        if (HtmlTable.inert(tag)) return;
        if (tag.equals("br")) {
            writer.breakLine();
            return;
        }
        if (tag.equals("img")) {
            writer.text(element.attr("alt"));
            warnings.add(
                    "Markdown images retain textual references/alt text; image downloading, decoding and OCR"
                            + " are unavailable.");
            return;
        }
        if (tag.equals("pre") || tag.equals("textarea")) {
            writer.pre = true;
            for (Node child : element.childNodes()) walk(child, writer, out, layout, trim, warnings);
            writer.pre = false;
            return;
        }
        if (tag.equals("table")) {
            writer.flush(out, "text", true);
            emitTable(element, out, warnings);
            return;
        }
        if (!writer.pre && BLOCKS.contains(tag) && !writer.value.isEmpty() && !writer.endsNewline)
            writer.breakLine();
        for (Node child : element.childNodes()) walk(child, writer, out, layout, trim, warnings);
        if (!writer.pre && BLOCKS.contains(tag) && !writer.value.isEmpty() && !writer.endsNewline)
            writer.breakLine();
    }

    private static void emitTable(
            Element table, List<Map<String, Object>> out, Set<String> warnings) {
        // htmlTableImages strips image markup before TableData construction; preserve alt text for
        // unresolved media.
        Element copy = table.clone();
        for (Element image : copy.select("img")) {
            String src = image.attr("src");
            String lower = src.strip().toLowerCase(Locale.ROOT);
            String text = image.attr("alt");
            if (!lower.startsWith("data:")
                    && !lower.startsWith("http://")
                    && !lower.startsWith("https://")) text = "![" + text + "](" + src + ")";
            image.replaceWith(new TextNode(text));
            warnings.add(
                    "Markdown images retain textual references/alt text; image downloading, decoding and OCR"
                            + " are unavailable.");
        }
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("doc_type_kwd", "table");
        item.put("text", HtmlTable.render(copy));
        out.add(item);
    }

    private static String layout(String tag) {
        return switch (tag) {
            case "h1", "h2", "h3", "h4", "h5", "h6" -> "heading";
            case "p" -> "paragraph";
            case "ul", "ol", "li" -> "list";
            case "pre", "code" -> "code";
            case "blockquote" -> "quote";
            default -> "";
        };
    }

    private static final class Writer {
        StringBuilder value = new StringBuilder();
        boolean lastSpace, lineStart, endsNewline, pre;

        void text(String text) {
            for (int point : text.codePoints().toArray()) {
                if (pre) {
                    value.appendCodePoint(point);
                    endsNewline = point == '\n';
                    lastSpace = false;
                    lineStart = false;
                    continue;
                }
                if (point == ' ' || point == '\t' || point == '\n' || point == '\r' || point == '\f') {
                    if (lineStart || lastSpace) continue;
                    value.append(' ');
                    lastSpace = true;
                    lineStart = false;
                    endsNewline = false;
                } else {
                    value.appendCodePoint(point);
                    lastSpace = false;
                    lineStart = false;
                    endsNewline = false;
                }
            }
        }

        void breakLine() {
            if (!pre && lastSpace && !value.isEmpty()) value.setLength(value.length() - 1);
            value.append('\n');
            lastSpace = false;
            lineStart = true;
            endsNewline = true;
        }

        void flush(List<Map<String, Object>> out, String layout, boolean trim) {
            String text = trim ? value.toString().strip() : value.toString();
            if (text.isEmpty()) return;
            out.add(GeneralChunker.unit(text, layout));
            value.setLength(0);
            lastSpace = false;
            lineStart = true;
            endsNewline = false;
        }
    }
}
