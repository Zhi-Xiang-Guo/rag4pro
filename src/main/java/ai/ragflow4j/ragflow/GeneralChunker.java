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
import java.util.regex.Pattern;

/**
 * 将解析单元合并为可检索的文本块，对应 Go ingestion/component/chunker/general.go。
 *
 * <p>普通文本和 Markdown 有各自的合并规则；token 目标控制合并时机，不能视为所有块的硬上限。 如配置子块分隔符，则在合并后再拆成子块，并保留父块正文供聊天回填。
 */
public final class GeneralChunker {
    public record Config(
            int chunkTokenSize,
            List<String> delimiters,
            List<String> childrenDelimiters,
            double overlapPercent) {
        public Config {
            if (chunkTokenSize < 0)
                throw new IllegalArgumentException("chunk_token_size must be non-negative");
            if (!Double.isFinite(overlapPercent))
                throw new IllegalArgumentException("overlap must be finite");
            delimiters = List.copyOf(delimiters);
            childrenDelimiters = List.copyOf(childrenDelimiters);
        }

        public static Config defaults() {
            return new Config(512, List.of("\n", "!", "?", ";", "。", "；", "！", "？"), List.of(), 0);
        }
    }

    private record Part(Map<String, Object> unit, boolean blockStart, boolean shortHeading) {
    }

    private final ContextBuilder.TokenCounter tokens;

    public GeneralChunker(ContextBuilder.TokenCounter tokens) {
        this.tokens = Objects.requireNonNull(tokens);
    }

    public List<Map<String, Object>> chunkText(String text, Config config) {
        return chunk(List.of(unit(normalize(text), "")), config, false);
    }

    public List<Map<String, Object>> chunkMarkdown(
            List<Map<String, Object>> parserUnits, Config config) {
        return chunk(parserUnits, config, true);
    }

    public List<Map<String, Object>> chunk(
            List<Map<String, Object>> parserUnits, Config config, boolean markdown) {
        // 先按主分隔符得到单元，再依据文本类型合并；分块预算使用 Cl100k，而非词法分词器。
        Pattern primary = compileDelimiters(config.delimiters());
        boolean keep = config.delimiters().stream().noneMatch(GeneralChunker::custom);
        List<Part> parts = new ArrayList<>();
        for (Map<String, Object> source : parserUnits) {
            Map<String, Object> sourceCopy = new LinkedHashMap<>(source);
            String text = normalize(text(source));
            if (markdown && (!keep || heading(source))) text = text.strip();
            if (!markdown) clearLayout(sourceCopy);
            boolean shortHeading =
                    markdown
                            && heading(source)
                            && tokens.count(text.replaceFirst("^#{1,6} +", "").strip()) < 50;
            List<String> fragments =
                    type(source).equals("text") ? split(text, primary, keep, false) : List.of(text);
            if (keep) fragments = foldWhitespace(fragments);
            boolean starts = true;
            for (String fragment : fragments) {
                if (!keep && fragment.isBlank()) continue;
                if (markdown && !keep) fragment = fragment.strip();
                Map<String, Object> piece = new LinkedHashMap<>(sourceCopy);
                piece.put("text", fragment);
                piece.put("tk_nums", tokens.count(fragment));
                parts.add(new Part(piece, starts, shortHeading));
                starts = !keep;
            }
        }
        List<Map<String, Object>> merged;
        if (markdown) merged = mergeMarkdown(parts, config);
        else if (keep) merged = mergeText(parts, config);
        else merged = parts.stream().map(Part::unit).toList();
        return finalizeChunks(merged, compileDelimiters(config.childrenDelimiters()));
    }

    private List<Map<String, Object>> mergeText(List<Part> parts, Config config) {
        double overlap = Math.max(0, Math.min(100, config.overlapPercent()));
        double threshold = config.chunkTokenSize() * (100 - overlap) / 100;
        List<Map<String, Object>> chunks = new ArrayList<>();
        Map<String, Object> current = null;
        for (Part part : parts) {
            Map<String, Object> unit = part.unit();
            if (!type(unit).equals("text")) {
                chunks.add(unit);
                current = null;
                continue;
            }
            if (text(unit).isBlank()) continue;
            if (count(unit) > config.chunkTokenSize()) {
                chunks.add(unit);
                current = null;
                continue;
            }
            // 判断的是追加前的当前块；因此追加完整单元后可以超过目标，长单元也整体保留。
            if (current == null || count(current) > threshold) {
                chunks.add(unit);
                current = unit;
            } else merge(current, unit, "");
        }
        Map<String, Object> previous = null;
        for (Map<String, Object> chunk : chunks) {
            if (!type(chunk).equals("text")) {
                previous = null;
                continue;
            }
            if (previous != null && overlap > 0) {
                chunk.put("text", overlap(text(previous), overlap) + text(chunk));
                chunk.put("tk_nums", tokens.count(text(chunk)));
            }
            previous = chunk;
        }
        return chunks;
    }

    private List<Map<String, Object>> mergeMarkdown(List<Part> parts, Config config) {
        List<Map<String, Object>> chunks = new ArrayList<>();
        Map<String, Object> current = null;
        for (int i = 0; i < parts.size(); i++) {
            Part part = parts.get(i);
            Map<String, Object> unit = part.unit();
            boolean previousShort = i > 0 && parts.get(i - 1).shortHeading();
            String separator =
                    part.blockStart()
                            && current != null
                            && !text(current).endsWith("\n")
                            && !text(unit).startsWith("\n")
                            ? "\n"
                            : "";
            if (type(unit).equals("table")) {
                if (current != null && heading(current) && previousShort) {
                    Map<String, Object> table = new LinkedHashMap<>(unit);
                    table.put("text", text(current) + separator + text(unit));
                    table.put("tk_nums", count(current) + count(unit));
                    chunks.set(chunks.size() - 1, table);
                } else chunks.add(unit);
                current = null;
                continue;
            }
            unit.put("doc_type_kwd", "text");
            if (current == null || part.blockStart() && heading(unit) && !heading(current)) {
                chunks.add(unit);
                current = unit;
                continue;
            }
            int projected = count(current) + count(unit);
            boolean force = heading(current) && previousShort;
            if (!force && projected > config.chunkTokenSize()) {
                String prefix = overlap(text(current), Math.max(0, Math.min(100, config.overlapPercent())));
                if (!prefix.isEmpty()) {
                    unit.put("text", joinOverlap(prefix, text(unit), separator));
                    unit.put("tk_nums", tokens.count(prefix) + count(unit));
                }
                chunks.add(unit);
                current = unit;
                continue;
            }
            merge(current, unit, separator);
            if (part.blockStart() || !heading(unit)) clearLayout(current);
        }
        return chunks;
    }

    /**
     * 完成父子块拆分：子块携带父块正文 mom，后续 materialize 再将它换成 mom_id。
     */
    private List<Map<String, Object>> finalizeChunks(
            List<Map<String, Object>> chunks, Pattern children) {
        List<Map<String, Object>> result = new ArrayList<>();
        for (Map<String, Object> source : chunks) {
            Map<String, Object> chunk = new LinkedHashMap<>(source);
            String text = removeTags(text(chunk));
            if (text.isBlank()) continue;
            chunk.put("text", text);
            if (children == null || !type(chunk).equals("text")) {
                chunk.put("tk_nums", tokens.count(text));
                result.add(chunk);
                continue;
            }
            String mom = text.startsWith("\n") ? text.substring(1) : text;
            List<String> parts =
                    split(text, children, true, true).stream().filter(p -> !p.isBlank()).toList();
            if (parts.isEmpty()) {
                chunk.put("tk_nums", tokens.count(text));
                result.add(chunk);
                continue;
            }
            for (String part : parts) {
                Map<String, Object> child = new LinkedHashMap<>(chunk);
                child.put("text", part);
                child.put("mom", mom);
                child.put("tk_nums", tokens.count(part));
                result.add(child);
            }
        }
        return result;
    }

    public static Map<String, Object> unit(String text, String layout) {
        Map<String, Object> unit = new LinkedHashMap<>();
        unit.put("text", text);
        unit.put("doc_type_kwd", "text");
        if (!layout.isEmpty()) unit.put("layout", layout);
        return unit;
    }

    static String normalize(String text) {
        return text.replace("\r\n", "\n").replace('\r', '\n');
    }

    private static String text(Map<String, Object> unit) {
        return Objects.toString(unit.get("text"), "");
    }

    private static String type(Map<String, Object> unit) {
        return Objects.toString(unit.get("doc_type_kwd"), "text");
    }

    private static boolean heading(Map<String, Object> unit) {
        return "heading".equals(unit.get("layout"));
    }

    private static int count(Map<String, Object> unit) {
        return ((Number) unit.get("tk_nums")).intValue();
    }

    private static void clearLayout(Map<String, Object> unit) {
        for (String key : List.of("layout", "layout_type", "layout_no")) unit.remove(key);
    }

    private static void merge(
            Map<String, Object> current, Map<String, Object> unit, String separator) {
        current.put("text", text(current) + separator + text(unit));
        current.put("tk_nums", count(current) + count(unit));
        unit.forEach(current::putIfAbsent);
    }

    public static Pattern compileDelimiters(List<String> delimiters) {
        Set<String> distinct = new LinkedHashSet<>();
        for (String delimiter : delimiters) {
            if (custom(delimiter)) delimiter = delimiter.substring(1, delimiter.length() - 1);
            if (!delimiter.isEmpty()) distinct.add(delimiter);
        }
        List<String> ordered = new ArrayList<>(distinct);
        ordered.sort(Comparator.comparingInt((String s) -> s.codePointCount(0, s.length())).reversed());
        return ordered.isEmpty()
                ? null
                : Pattern.compile(String.join("|", ordered.stream().map(Pattern::quote).toList()));
    }

    private static boolean custom(String delimiter) {
        return delimiter.length() >= 2 && delimiter.startsWith("`") && delimiter.endsWith("`");
    }

    private static List<String> split(String text, Pattern pattern, boolean keep, boolean children) {
        if (pattern == null) return List.of(text);
        var matcher = pattern.matcher(text);
        List<String> result = new ArrayList<>();
        int cursor = 0;
        boolean found = false;
        while (matcher.find()) {
            found = true;
            if (!children && keep || matcher.start() > cursor)
                result.add(text.substring(cursor, keep ? matcher.end() : matcher.start()));
            cursor = matcher.end();
        }
        if (!found || cursor < text.length()) result.add(text.substring(cursor));
        return result;
    }

    private static List<String> foldWhitespace(List<String> pieces) {
        List<String> result = new ArrayList<>();
        String pending = "";
        for (String piece : pieces) {
            if (piece.isBlank()) {
                pending += piece;
                continue;
            }
            result.add(pending + piece);
            pending = "";
        }
        if (!pending.isEmpty() && !result.isEmpty())
            result.set(result.size() - 1, result.getLast() + pending);
        return result;
    }

    private static String overlap(String previous, double percent) {
        String visible = removeTags(previous);
        int length = visible.codePointCount(0, visible.length());
        int cut = Math.max(0, Math.min(length, (int) (length * (100 - percent) / 100)));
        return visible.substring(visible.offsetByCodePoints(0, cut));
    }

    private static String joinOverlap(String prefix, String text, String separator) {
        if (prefix.isEmpty()) return text;
        if (text.isEmpty()
                || separator.isEmpty()
                || prefix.endsWith(separator)
                || text.startsWith(separator)) return prefix + text;
        return prefix + separator + text;
    }

    private static String removeTags(String text) {
        return text.replaceAll("@@[\\t0-9.-]+?##", "");
    }
}
