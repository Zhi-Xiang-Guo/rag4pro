package ai.ragflow4j.ragflow;

import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/** The same chunk plan drives previews, single imports and batch imports. */
final class ChunkingStrategies {
    private static final Set<String> STRATEGIES =
            Set.of("general", "fixed", "section", "parent_child", "hierarchical");
    private final Cl100k tokens;
    private final GeneralChunker general;

    record Plan(String strategy, GeneralChunker.Config config, List<Map<String, Object>> pieces,
                List<String> warnings) {}

    private record Section(String heading, List<String> paragraphs) {}

    ChunkingStrategies(Cl100k tokens) {
        this.tokens = tokens;
        general = new GeneralChunker(tokens);
    }

    static void validate(String strategy, GeneralChunker.Config config) {
        if (!STRATEGIES.contains(strategy)) throw new IllegalArgumentException("未知分块策略");
        if (strategy.equals("general")) return;
        if (config.chunkTokenSize() < 32 || config.chunkTokenSize() > 1024)
            throw new IllegalArgumentException("分块大小须为 32–1024 tokens");
        if (config.overlapPercent() < 0 || config.overlapPercent() > 50
                || (!strategy.equals("fixed") && config.overlapPercent() != 0))
            throw new IllegalArgumentException("固定窗口支持 0–50% 重叠，其他策略按段落边界分块");
    }

    Plan plan(String text, String format, String strategy, GeneralChunker.Config config) {
        validate(strategy, config);
        boolean markdown = switch (format) {
            case "md" -> true;
            case "txt" -> false;
            default -> throw new IllegalArgumentException("支持 .txt 和 .md 文件");
        };
        if (strategy.equals("fixed")) {
            List<Map<String, Object>> pieces = windows(GeneralChunker.normalize(text),
                    config.chunkTokenSize(), config.overlapPercent()).stream()
                    .map(value -> leaf(value, "", strategy)).toList();
            return new Plan(strategy, config, pieces, List.of());
        }
        var parsed = markdown ? new MarkdownParser().parse(text) : null;
        List<String> warnings = markdown ? parsed.warnings() : List.of();
        if (strategy.equals("general")) {
            return new Plan(strategy, config, markdown ? general.chunkMarkdown(parsed.units(), config)
                    : general.chunkText(text, config), warnings);
        }
        List<Section> sections = markdown ? markdownSections(parsed.units())
                : List.of(new Section("", List.of(GeneralChunker.normalize(text).split("\\R\\s*\\R"))));
        return new Plan(strategy, config, structured(sections, strategy, config.chunkTokenSize()), warnings);
    }

    private List<Section> markdownSections(List<Map<String, Object>> units) {
        List<Section> sections = new ArrayList<>();
        List<String> paragraphs = new ArrayList<>();
        TreeMap<Integer, String> headings = new TreeMap<>();
        for (var unit : units) {
            String text = ContextBuilder.text(unit, "text").strip();
            if (text.isEmpty()) continue;
            if ("heading".equals(unit.get("layout"))) {
                if (!paragraphs.isEmpty()) {
                    sections.add(new Section(String.join(" / ", headings.values()), paragraphs));
                    paragraphs = new ArrayList<>();
                }
                int depth = text.indexOf(' ');
                if (depth < 1) continue;
                headings.tailMap(depth, true).clear();
                headings.put(depth, text.substring(depth + 1));
            } else paragraphs.add(text);
        }
        if (!paragraphs.isEmpty())
            sections.add(new Section(String.join(" / ", headings.values()), paragraphs));
        if (sections.isEmpty() && !headings.isEmpty())
            sections.add(new Section("", List.copyOf(headings.values())));
        return sections;
    }

    private List<Map<String, Object>> structured(List<Section> sections, String strategy, int size) {
        List<Map<String, Object>> pieces = new ArrayList<>();
        int parentNumber = 0, sectionNumber = 0;
        for (Section section : sections) {
            List<String> leaves = new ArrayList<>();
            for (String paragraph : section.paragraphs()) {
                if (paragraph.isBlank()) continue;
                leaves.addAll(packParagraph(paragraph.strip(), size));
            }
            if (strategy.equals("section")) {
                for (String value : leaves) pieces.add(leaf(value, section.heading(), strategy));
                continue;
            }
            List<List<String>> parents = group(leaves, Math.min(size * 4, 2048));
            // A section window collects adjacent parents under the same Markdown heading.
            for (int first = 0; first < parents.size(); first += 2) {
                var window = parents.subList(first, Math.min(first + 2, parents.size()));
                String sectionText = strategy.equals("hierarchical")
                        ? String.join("\n\n", window.stream().map(parts -> String.join("\n\n", parts)).toList())
                        : "";
                for (var parent : window) {
                    String parentText = String.join("\n\n", parent);
                    for (String value : parent) {
                        var piece = leaf(value, section.heading(), strategy);
                        piece.put("mom", parentText);
                        piece.put("parent_ref", parentNumber);
                        if (!sectionText.isEmpty()) {
                            piece.put("section_context", sectionText);
                            piece.put("section_ref", sectionNumber);
                        }
                        pieces.add(piece);
                    }
                    parentNumber++;
                }
                sectionNumber++;
            }
        }
        return pieces;
    }

    private List<String> packParagraph(String text, int size) {
        if (tokens.count(text) <= size) return List.of(text);
        var sentences = BreakIterator.getSentenceInstance(Locale.ROOT);
        sentences.setText(text);
        List<String> parts = new ArrayList<>();
        for (int start = sentences.first(), end = sentences.next(); end != BreakIterator.DONE;
             start = end, end = sentences.next()) {
            String sentence = text.substring(start, end).strip();
            parts.addAll(windows(sentence, size, 0));
        }
        return group(parts, size).stream().map(group -> String.join("\n\n", group)).toList();
    }

    private List<List<String>> group(List<String> parts, int limit) {
        List<List<String>> assembled = new ArrayList<>();
        List<String> current = new ArrayList<>();
        String currentText = "";
        for (String part : parts) {
            String joined = current.isEmpty() ? part : currentText + "\n\n" + part;
            if (!current.isEmpty() && tokens.count(joined) > limit) {
                assembled.add(current);
                current = new ArrayList<>();
                joined = part;
            }
            current.add(part);
            currentText = joined;
        }
        if (!current.isEmpty()) assembled.add(current);
        return assembled;
    }

    private List<String> windows(String text, int size, double overlap) {
        List<String> parts = new ArrayList<>();
        int overlapTokens = (int) (size * overlap / 100);
        for (int start = 0; start < text.length(); ) {
            int end = Math.min(text.length(), start + size * 12);
            if (end < text.length() && Character.isLowSurrogate(text.charAt(end))) end--;
            String part = tokens.trim(text.substring(start, end), size);
            if (part.isEmpty()) part = text.substring(start, text.offsetByCodePoints(start, 1));
            if (!part.isBlank()) parts.add(part);
            if (start + part.length() == text.length()) break;
            int advance = overlapTokens == 0 ? part.length()
                    : tokens.trim(part, Math.max(1, tokens.count(part) - overlapTokens)).length();
            start += advance == 0 ? part.length() : advance;
        }
        return parts;
    }

    private Map<String, Object> leaf(String text, String heading, String strategy) {
        var unit = GeneralChunker.unit(text, "");
        unit.put("tk_nums", tokens.count(text));
        unit.put("chunk_strategy_kwd", strategy);
        if (!heading.isEmpty()) {
            unit.put("section_kwd", heading);
            // The heading helps both lexical and vector matching without changing the quoted text.
            unit.put("context_above", heading + "\n");
        }
        return unit;
    }
}
