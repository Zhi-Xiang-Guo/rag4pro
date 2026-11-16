// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations.
package ai.ragflow4j.ragflow;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;

/**
 * 将回答中的引用标记连接到本次检索证据，对应 citation.go 和 chat_pipeline.go 的 decorateAnswer。
 *
 * <p>[ID:n] 的 n 是当前知识块数组下标，和持久化的 chunk_id 不同。 修复引用格式或按语义相似度补引都不等于验证回答事实，仍需核对证据正文。
 */
public final class CitationService {
    private static final String DIGITS = "[0-9\\x{0660}-\\x{0669}\\x{06F0}-\\x{06F9}]+";
    private static final Pattern MARKER = Pattern.compile("\\[(?:ID:\\s*)?(" + DIGITS + ")\\]");
    private static final Pattern SENTENCE =
            Pattern.compile("([^\\|][；。？!！,؛؟.\\n]|[a-z؀-ۿ][.?;!،؛؟][ \\n])");
    private static final List<Pattern> BAD =
            List.of(
                    Pattern.compile("\\(\\s*ID\\s*[:： ]*\\s*(" + DIGITS + ")\\s*\\)"),
                    Pattern.compile("\\[\\s*ID\\s*[:： ]*\\s*(" + DIGITS + ")\\s*\\]"),
                    Pattern.compile("【\\s*ID\\s*[:： ]*\\s*(" + DIGITS + ")\\s*】"),
                    Pattern.compile("(?i)\\bref\\s*(" + DIGITS + ")\\b"),
                    Pattern.compile("\\(\\s*\\**\\s*ID\\s*[:： ]*\\s*(" + DIGITS + ")\\s*\\**\\s*\\)"));

    @FunctionalInterface
    public interface Embedder {
        List<double[]> encode(List<String> texts) throws Exception;
    }

    @FunctionalInterface
    public interface VectorHydrator {
        /**
         * 为传入的块副本补读向量，保持数组顺序，避免改变引用下标。
         */
        void hydrate(List<Map<String, Object>> chunks, int dimension) throws Exception;
    }

    public record Inserted(String answer, List<Integer> citedIndices) {
    }

    public record Decorated(
            String answer, Map<String, Object> reference, List<Integer> citedIndices) {
    }

    private CitationService() {
    }

    public static String repair(String answer) {
        String result = answer;
        for (Pattern pattern : BAD) {
            result = pattern.matcher(result).replaceAll(m -> "[ID:" + normalizeDigits(m.group(1)) + "]");
        }
        return result;
    }

    public static String normalizeDigits(String text) {
        StringBuilder result = new StringBuilder();
        text.codePoints()
                .forEach(
                        c -> {
                            if (c >= 0x0660 && c <= 0x0669) c = c - 0x0660 + '0';
                            else if (c >= 0x06f0 && c <= 0x06f9) c = c - 0x06f0 + '0';
                            result.appendCodePoint(c);
                        });
        return result.toString();
    }

    public static boolean hasMarkers(String text) {
        return MARKER.matcher(normalizeDigits(text)).find();
    }

    public static List<Integer> extract(String text, int maxIndex) {
        Set<Integer> indices = new LinkedHashSet<>();
        var matcher = MARKER.matcher(normalizeDigits(text));
        while (matcher.find()) {
            try {
                int index = Integer.parseInt(matcher.group(1));
                if (index >= 0 && index < maxIndex) indices.add(index);
            } catch (NumberFormatException ignored) {
                /* 越界编号无法指向当前知识块，忽略它。 */
            }
        }
        return List.copyOf(indices);
    }

    public static String cleanHistory(String text) {
        return text.replaceAll("##[0-9]+\\$\\$", "");
    }

    public static String strip(String text) {
        String result =
                cleanHistory(text)
                        .replaceAll("(?i)\\[\\s*ID\\s*[:： ]*\\s*" + DIGITS + "\\s*\\]", "")
                        .replaceAll("(?i)\\[\\s*(?:ID\\s*[:： ]*\\s*)?slot\\s*[0-9]+\\s*\\]", "")
                        .replaceAll("(?i)\\[\\s*ID\\s*[:： ]*\\s*[0-9]+\\s*[-–—~～]\\s*[0-9]+\\s*\\]", "");
        for (Pattern pattern : BAD) result = pattern.matcher(result).replaceAll("");
        return result;
    }

    /**
     * 将回答拆句并与知识块向量比较，沿用原版句子重组、UTF-8 长度检查和全局阈值下降规则。
     */
    public static Inserted insert(String answer, List<double[]> chunkVectors, Embedder embedder)
            throws Exception {
        List<String> pieces = pieces(answer);
        List<String> sentences = new ArrayList<>();
        List<Integer> positions = new ArrayList<>();
        for (int i = 0; i < pieces.size(); i++) {
            if (pieces.get(i).strip().getBytes(StandardCharsets.UTF_8).length >= 5) {
                sentences.add(pieces.get(i));
                positions.add(i);
            }
        }
        if (sentences.isEmpty() || chunkVectors.isEmpty()) return new Inserted(answer, List.of());
        List<double[]> encoded = embedder.encode(sentences);
        if (encoded.isEmpty()) return new Inserted(answer, List.of());
        int count = Math.min(sentences.size(), encoded.size());
        double[][] scores = new double[count][chunkVectors.size()];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < chunkVectors.size(); j++)
                scores[i][j] = cosine(encoded.get(i), chunkVectors.get(j));
        }
        Map<Integer, List<Integer>> citations = new LinkedHashMap<>();
        // 仅当这一轮所有句子都没有引用时才降低阈值；不是每个句子单独不断放宽。
        for (double threshold = 0.63; threshold > 0.3 && citations.isEmpty(); threshold *= 0.8) {
            for (int i = 0; i < count; i++) {
                double max = Arrays.stream(scores[i]).max().orElse(0) * 0.99;
                if (max < threshold) continue;
                List<Integer> matches = new ArrayList<>();
                for (int j = 0; j < scores[i].length && matches.size() < 4; j++) {
                    if (scores[i][j] > max) matches.add(j);
                }
                if (!matches.isEmpty()) citations.put(positions.get(i), matches);
            }
        }
        Set<Integer> seen = new LinkedHashSet<>();
        StringBuilder decorated = new StringBuilder();
        for (int i = 0; i < pieces.size(); i++) {
            decorated.append(pieces.get(i));
            for (int index : citations.getOrDefault(i, List.of())) {
                if (seen.add(index)) decorated.append(" [ID:").append(index).append(']');
            }
        }
        return new Inserted(decorated.toString(), List.copyOf(seen));
    }

    private static List<String> pieces(String answer) {
        List<String> pieces = new ArrayList<>();
        String[] blocks = answer.split("```", -1);
        for (int i = 0; i < blocks.length; i++) {
            if (i % 2 == 1) {
                pieces.add("```" + blocks[i] + "```\n");
            } else {
                var matcher = SENTENCE.matcher(blocks[i]);
                int start = 0;
                boolean matched = false;
                while (matcher.find()) {
                    pieces.add(blocks[i].substring(start, matcher.end()));
                    start = matcher.end();
                    matched = true;
                }
                if (!matched || start < blocks[i].length()) pieces.add(blocks[i].substring(start));
            }
        }
        for (int i = 1; i < pieces.size(); i++) {
            String piece = pieces.get(i);
            if (SENTENCE.matcher(piece).find()) {
                int end = piece.offsetByCodePoints(0, 1);
                pieces.set(i - 1, pieces.get(i - 1) + piece.substring(0, end));
                pieces.set(i, piece.substring(end));
            }
        }
        return pieces;
    }

    private static double cosine(double[] left, double[] right) {
        double dot = 0, a = 0, b = 0;
        for (double value : left) a += value * value;
        for (double value : right) b += value * value;
        for (int i = 0; i < Math.min(left.length, right.length); i++) dot += left[i] * right[i];
        return a == 0 || b == 0 ? 0 : dot / Math.sqrt(a) / Math.sqrt(b);
    }

    private static double[] vector(Map<String, Object> chunk) {
        Object value = chunk.get("vector");
        if (value instanceof double[] array) return array;
        if (value instanceof List<?> list) {
            return list.stream().mapToDouble(n -> ((Number) n).doubleValue()).toArray();
        }
        return new double[0];
    }

    /**
     * 先修复已有引用；缺少标记时尝试语义补引，最后输出证据块及被引用文档的聚合信息。
     */
    public static Decorated decorate(
            String answer,
            List<Map<String, Object>> sourceChunks,
            List<Map<String, Object>> docAggs,
            boolean hasKnowledge,
            boolean quote,
            Embedder embedder,
            VectorHydrator hydrator) {
        List<Map<String, Object>> chunks =
                sourceChunks.stream().map(c -> (Map<String, Object>) new LinkedHashMap<>(c)).toList();
        String think = "";
        String text = answer;
        String[] parts = answer.split("</think>", -1);
        if (parts.length == 2) {
            think = parts[0] + "</think>";
            text = parts[1].strip();
        }
        List<Integer> cited = List.of();
        if (!quote) {
            think = strip(think);
            text = strip(text);
        }
        if (hasKnowledge && quote && !chunks.isEmpty()) {
            think = repair(think);
            text = repair(text);
            String normalized = normalizeDigits(think + text);
            if (embedder != null && !hasMarkers(normalized)) {
                int dimension =
                        chunks.stream()
                                .mapToInt(c -> vector(c).length)
                                .filter(n -> n > 0)
                                .findFirst()
                                .orElse(0);
                try {
                    if (dimension == 0) {
                        List<double[]> probe = embedder.encode(List.of("x"));
                        if (!probe.isEmpty()) dimension = probe.getFirst().length;
                    }
                    if (hydrator != null) hydrator.hydrate(chunks, dimension);
                } catch (Exception ignored) {
                    /* 补读失败时保留已有向量，继续原版降级路径。 */
                }
            }
            if (embedder != null && !hasMarkers(text)) {
                int dimension =
                        chunks.stream()
                                .mapToInt(c -> vector(c).length)
                                .filter(n -> n > 0)
                                .findFirst()
                                .orElse(0);
                if (dimension > 0) {
                    List<double[]> vectors = new ArrayList<>();
                    for (Map<String, Object> chunk : chunks) {
                        double[] value = vector(chunk);
                        vectors.add(value.length == dimension ? value : new double[dimension]);
                    }
                    try {
                        Inserted result = insert(text, vectors, embedder);
                        if (!result.citedIndices().isEmpty()) {
                            text = result.answer();
                            cited = result.citedIndices();
                        }
                    } catch (Exception ignored) {
                        /* 补引 embedding 失败时保留模型原回答。 */
                    }
                }
            } else if (hasMarkers(normalized)) cited = extract(normalized, chunks.size());
        }
        Map<String, Object> reference = new LinkedHashMap<>();
        if (hasKnowledge) {
            reference.put("total", chunks.size());
            reference.put("chunks", chunks.stream().map(CitationService::formatChunk).toList());
            if (quote) {
                Set<String> citedDocs = new HashSet<>();
                for (int index : cited) {
                    String doc = ContextBuilder.text(chunks.get(index), "doc_id");
                    if (!doc.isEmpty()) citedDocs.add(doc);
                }
                List<Map<String, Object>> filtered = docAggs;
                if (!citedDocs.isEmpty())
                    filtered = docAggs.stream().filter(d -> citedDocs.contains(d.get("doc_id"))).toList();
                reference.put("doc_aggs", filtered);
            }
        }
        if (text.toLowerCase(Locale.ROOT).contains("invalid key")
                || text.toLowerCase(Locale.ROOT).contains("invalid api")) {
            text += " Please set LLM API-Key in 'User Setting -> Model providers -> API-Key'";
        }
        return new Decorated(think + text, reference, cited);
    }

    public static Map<String, Object> formatChunk(Map<String, Object> chunk) {
        Map<String, Object> result = new LinkedHashMap<>();
        String[][] aliases = {
                {"id", "chunk_id", "id"},
                {"content", "content", "content_with_weight"},
                {"document_id", "doc_id", "document_id"},
                {"document_name", "docnm_kwd", "document_name"},
                {"dataset_id", "kb_id", "dataset_id"},
                {"image_id", "image_id", "img_id"},
                {"positions", "positions", "position_int"},
                {"doc_type", "doc_type_kwd", "doc_type"}
        };
        for (String[] alias : aliases) {
            result.put(alias[0], chunk.containsKey(alias[1]) ? chunk.get(alias[1]) : chunk.get(alias[2]));
        }
        for (String key :
                List.of(
                        "url",
                        "similarity",
                        "vector_similarity",
                        "term_similarity",
                        "row_id",
                        "document_external_id",
                        "document_metadata")) {
            result.put(key, chunk.get(key));
        }
        for (String key : List.of("section_kwd", "context_level_kwd", "chunk_strategy_kwd", "score_type",
                "mom_id", "section_id"))
            if (chunk.containsKey(key)) result.put(key, chunk.get(key));
        if (chunk.containsKey("source_revision_kwd")) {
            var location = new LinkedHashMap<String, Object>();
            location.put("revision", chunk.get("source_revision_kwd"));
            location.put("kind", chunk.get("source_kind_kwd"));
            location.put("unit", chunk.get("source_unit_int"));
            if (chunk.containsKey("source_start_int")) location.put("start", chunk.get("source_start_int"));
            result.put("source", location);
        }
        return result;
    }
}
