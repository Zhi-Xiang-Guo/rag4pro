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

/**
 * 把分块结果加工为检索字段：粗细分词、标题词、正文向量和标题混合向量。
 *
 * <p>对应 tokenizer.go；本类返回加工后的块，由 CorpusService 补齐文档标识后写入 ES。
 */
public final class IndexingPipeline {
    public interface LexicalTokenizer {
        String tokenize(String text, String language);

        String fineGrainedTokenize(String tokens, String language);
    }

    public interface Embedder {
        List<double[]> embed(List<String> texts) throws Exception;

        default int maxTokens() {
            return 2048;
        }

        default int batchSize() {
            return 16;
        }

        /**
         * 模型可提供自己的 token 裁剪器；返回 null 时使用通用计数器，对应 Go EmbedderTrimmer。
         */
        default String trim(String text) {
            return null;
        }
    }

    public record Config(
            String language,
            boolean fullText,
            boolean embedding,
            double filenameEmbeddingWeight,
            List<String> fields) {
        public Config {
            language = Objects.requireNonNullElse(language, "English");
            fields = fields == null ? List.of("text") : List.copyOf(fields);
            if (!Double.isFinite(filenameEmbeddingWeight)
                    || filenameEmbeddingWeight < 0
                    || filenameEmbeddingWeight > 1) {
                throw new IllegalArgumentException("filename embedding weight must be in [0,1]");
            }
        }

        public static Config defaults(String language) {
            return new Config(language, true, true, 0.1, List.of("text"));
        }
    }

    private static final Set<String> STOPWORDS =
            Set.of(
                    ("a an and are as at be been but by d for from had has have he her him his i if in into"
                            + " is it its ll m me my no not of on or our re s she so t that the their them then"
                            + " there these they this to too up us ve was we were what when where which who will"
                            + " with y you your title name fullname wikipedia")
                            .split(" "));
    private final LexicalTokenizer lexical;
    private final ContextBuilder.TokenCounter tokens;
    private final Embedder embedder;
    private final ContentVectorCache cache;
    private final String embeddingModelId;

    public IndexingPipeline(
            NativeTokenizer tokenizer, ContextBuilder.TokenCounter tokens, Embedder embedder) {
        this(tokenizer, tokens, embedder, null, "");
    }

    public IndexingPipeline(
            NativeTokenizer tokenizer,
            ContextBuilder.TokenCounter tokens,
            Embedder embedder,
            ContentVectorCache cache,
            String embeddingModelId) {
        this(
                new LexicalTokenizer() {
                    public String tokenize(String text, String language) {
                        return tokenizer.tokenize(text, language);
                    }

                    public String fineGrainedTokenize(String text, String language) {
                        return tokenizer.fineGrainedTokenize(text, language);
                    }
                },
                tokens,
                embedder,
                cache,
                embeddingModelId);
    }

    public IndexingPipeline(
            LexicalTokenizer lexical, ContextBuilder.TokenCounter tokens, Embedder embedder) {
        this(lexical, tokens, embedder, null, "");
    }

    public IndexingPipeline(
            LexicalTokenizer lexical,
            ContextBuilder.TokenCounter tokens,
            Embedder embedder,
            ContentVectorCache cache,
            String embeddingModelId) {
        this.lexical = Objects.requireNonNull(lexical);
        this.tokens = Objects.requireNonNull(tokens);
        this.embedder = embedder;
        this.cache = cache;
        this.embeddingModelId = Objects.requireNonNullElse(embeddingModelId, "");
    }

    public List<Map<String, Object>> process(
            String name, List<Map<String, Object>> input, Config config) throws Exception {
        return process(name, input, config, "");
    }

    public List<Map<String, Object>> process(
            String name, List<Map<String, Object>> input, Config config, String taskId) throws Exception {
        List<Map<String, Object>> chunks =
                input.stream()
                        .filter(
                                c ->
                                        !string(c, "text").isBlank()
                                                || !string(c, "context_above").isBlank()
                                                || !string(c, "context_below").isBlank())
                        .map(c -> (Map<String, Object>) new LinkedHashMap<>(c))
                        .toList();
        // 词法标题去掉文件扩展名并吸收声明标题；embedding 标题仍使用原始文件名 name。
        String titleStem = name.replaceFirst("\\.[a-zA-Z]+$", "");
        List<String> titles = declaredTitleTokens(chunks);
        if (!titles.isEmpty()) titleStem = (titleStem + " " + String.join(" ", titles)).strip();
        for (int i = 0; i < chunks.size(); i++) {
            Map<String, Object> chunk = chunks.get(i);
            chunk.put("chunk_order_int", i);
            if (config.fullText()) tokenize(chunk, titleStem, config.language());
        }
        if (config.embedding() && !chunks.isEmpty()) embed(name, chunks, config, taskId);
        return chunks;
    }

    /** 批量导入只合并模型请求；每篇文档仍走相同的分词、裁剪和标题混合算法。 */
    public record Document(String name, List<Map<String, Object>> pieces, Config config) {}

    public List<List<Map<String, Object>>> processBatch(List<Document> documents) throws Exception {
        LinkedHashSet<String> inputs = new LinkedHashSet<>();
        for (Document document : documents) {
            if (!document.config().embedding()) continue;
            if (!document.name().isBlank()) inputs.add(document.name());
            for (var chunk : document.pieces()) {
                String text = cleanEmbeddingText(concatFields(chunk, document.config().fields()));
                if (!text.isEmpty()) inputs.add(trimInput(text));
            }
        }
        List<String> texts = new ArrayList<>(inputs);
        if (!texts.isEmpty() && embedder.batchSize() <= 0)
            throw new IllegalArgumentException("embedder reported non-positive batch size");
        Map<String, double[]> vectors = new HashMap<>();
        // 限定本批生命周期，避免长期缓存吞掉堆内存，也不把测试标签存进索引。
        for (int i = 0; i < texts.size(); i += embedder.batchSize()) {
            List<String> batch = texts.subList(i, Math.min(i + embedder.batchSize(), texts.size()));
            List<double[]> encoded = embedder.embed(batch);
            checkVectors(encoded, batch.size());
            for (int j = 0; j < batch.size(); j++) vectors.put(batch.get(j), encoded.get(j));
        }
        Embedder prefetched = new Embedder() {
            public List<double[]> embed(List<String> batch) {
                return batch.stream().map(text -> Objects.requireNonNull(vectors.get(text),
                        "batch embedding plan mismatch")).toList();
            }
            public int maxTokens() { return embedder.maxTokens(); }
            public int batchSize() { return embedder.batchSize(); }
            public String trim(String text) { return embedder.trim(text); }
        };
        IndexingPipeline pipeline = new IndexingPipeline(lexical, tokens, prefetched);
        List<List<Map<String, Object>>> result = new ArrayList<>();
        for (Document document : documents)
            result.add(pipeline.process(document.name(), document.pieces(), document.config()));
        return result;
    }

    private String trimInput(String text) {
        String trimmed = embedder.trim(text);
        if (trimmed == null) trimmed = tokens.trim(text, embeddingTokenLimit(embedder.maxTokens()));
        return trimmed.isBlank() ? text : trimmed;
    }

    private void tokenize(Map<String, Object> chunk, String title, String language) {
        String titleTokens = lexical.tokenize(title, language);
        // *_tks/*_ltks 保存词法 token 字符串；*_sm_* 保存更细粒度的拆分结果。
        chunk.put("title_tks", titleTokens);
        chunk.put("title_sm_tks", lexical.fineGrainedTokenize(titleTokens, language));
        String questions = string(chunk, "questions");
        if (!questions.isEmpty()) {
            chunk.put("question_kwd", Arrays.asList(questions.split("\n", -1)));
            chunk.put("question_tks", lexical.tokenize(questions, language));
        }
        String keywords = string(chunk, "keywords");
        if (!keywords.isEmpty()) {
            chunk.put(
                    "important_kwd",
                    Arrays.stream(keywords.split(",", -1)).map(IndexingPipeline::sanitizeKeyword).toList());
            chunk.put("important_tks", lexical.tokenize(keywords, language));
        }
        String body = string(chunk, "summary").strip();
        if (body.isEmpty()) body = contextualText(chunk);
        if (!body.isBlank()) {
            String textTokens = lexical.tokenize(body, language);
            if (textTokens.isEmpty()) textTokens = body;
            chunk.put("content_ltks", textTokens);
            String fine = lexical.fineGrainedTokenize(textTokens, language);
            chunk.put("content_sm_ltks", fine.isEmpty() ? textTokens : fine);
        }
    }

    private void embed(String name, List<Map<String, Object>> chunks, Config config, String taskId)
            throws Exception {
        if (embedder == null)
            throw new IllegalStateException("Embedding requested without an embedder");
        List<String> texts = new ArrayList<>();
        List<Integer> positions = new ArrayList<>();
        double[][] contents = new double[chunks.size()][];
        boolean[] cacheHits = new boolean[chunks.size()];
        String[] cacheKeys = new String[chunks.size()];
        int hitCount = 0;
        for (int i = 0; i < chunks.size(); i++) {
            String text = cleanEmbeddingText(concatFields(chunks.get(i), config.fields()));
            if (text.isEmpty()) continue;
            String trimmed = trimInput(text);
            String chunkId = chunks.get(i).get("id") instanceof String id ? id : "";
            cacheKeys[i] = ContentVectorCache.key(embeddingModelId, chunkId, trimmed);
            if (cache != null) contents[i] = cache.get(cacheKeys[i]);
            if (contents[i] != null) {
                cacheHits[i] = true;
                hitCount++;
            } else {
                texts.add(trimmed);
                positions.add(i);
            }
        }
        if (texts.isEmpty() && hitCount == 0) return;
        double[] title = null;
        if (!name.isBlank()) {
            List<double[]> result = embedder.embed(List.of(name));
            checkVectors(result, 1);
            title = result.getFirst();
        }
        int batchSize = embedder.batchSize();
        if (batchSize <= 0)
            throw new IllegalArgumentException("embedder reported non-positive batch size");
        // 所有正文批次成功后才写缓存，避免后续批次失败时暴露本轮仅部分完成的正文向量。
        for (int start = 0; start < texts.size(); start += batchSize) {
            int end = Math.min(start + batchSize, texts.size());
            List<double[]> result = embedder.embed(texts.subList(start, end));
            checkVectors(result, end - start);
            for (int i = 0; i < result.size(); i++) contents[positions.get(start + i)] = result.get(i);
        }
        int dimension = title == null ? 0 : title.length;
        for (int i = 0; i < chunks.size(); i++) {
            double[] content = contents[i];
            if (content == null) continue;
            if (dimension == 0) dimension = content.length;
            if (content.length != dimension)
                throw new IllegalArgumentException("unexpected embedding dimensions");
            // 缓存保存原始正文向量；入索引时才混合标题，因此改变标题权重仍可复用正文缓存。
            // 混合后保持原始数值，不在这里额外做单位长度归一化。
            double[] vector = content.clone();
            if (title != null) {
                for (int k = 0; k < vector.length; k++)
                    vector[k] =
                            config.filenameEmbeddingWeight() * title[k]
                                    + (1 - config.filenameEmbeddingWeight()) * content[k];
            }
            chunks.get(i).put("q_" + vector.length + "_vec", vector);
            if (cache != null && !cacheHits[i]) cache.put(cacheKeys[i], content, taskId);
        }
    }

    private static void checkVectors(List<double[]> vectors, int expected) {
        if (vectors == null || vectors.size() != expected)
            throw new IllegalArgumentException("encode returned incorrect number of vectors");
        for (double[] vector : vectors) {
            if (vector == null || vector.length == 0)
                throw new IllegalArgumentException("empty embedding vector");
            for (double value : vector)
                if (!Double.isFinite(value))
                    throw new IllegalArgumentException("non-finite embedding vector");
        }
    }

    public static int embeddingTokenLimit(int maxTokens) {
        if (maxTokens <= 0) maxTokens = 2048;
        int margin = Math.max(32, (int) Math.ceil(maxTokens * 0.02));
        if (margin >= maxTokens) return maxTokens <= 1 ? maxTokens : maxTokens / 2;
        return maxTokens - margin;
    }

    public static String cleanEmbeddingText(String text) {
        return text.replaceAll("</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ").strip();
    }

    private static String concatFields(Map<String, Object> chunk, List<String> fields) {
        List<String> values = new ArrayList<>();
        for (String field : fields) {
            Object raw = field.equals("text") ? contextualText(chunk) : chunk.get(field);
            if (raw instanceof String value && !value.isEmpty()) values.add(value);
            else if (raw instanceof List<?> list && list.stream().allMatch(String.class::isInstance)) {
                values.add(String.join("\n", list.stream().map(String.class::cast).toList()));
            }
        }
        return String.join("", values);
    }

    private static String contextualText(Map<String, Object> chunk) {
        return string(chunk, "context_above") + string(chunk, "text") + string(chunk, "context_below");
    }

    private static String string(Map<String, Object> chunk, String key) {
        return Objects.toString(chunk.get(key), "");
    }

    private static String sanitizeKeyword(String word) {
        word = word.strip();
        int bytes = 0;
        int end = 0;
        while (end < word.length()) {
            int next = word.offsetByCodePoints(end, 1);
            int length = word.substring(end, next).getBytes(StandardCharsets.UTF_8).length;
            if (bytes + length > 32766) break;
            bytes += length;
            end = next;
        }
        return word.substring(0, end).stripTrailing();
    }

    public static List<String> declaredTitleTokens(List<Map<String, Object>> chunks) {
        List<String> values = new ArrayList<>();
        for (Map<String, Object> chunk : chunks) {
            String text = string(chunk, "text");
            if (text.isEmpty()) continue;
            String[] lines = text.split("\n", -1);
            for (int i = 0; i < Math.min(20, lines.length); i++) {
                String line = lines[i].strip().toLowerCase(Locale.ROOT);
                for (String key : List.of("title:", "name:", "fullname:")) {
                    if (line.startsWith(key)) values.add(line.substring(key.length()));
                }
            }
            break;
        }
        Set<String> accepted = new LinkedHashSet<>();
        for (String value : values) {
            for (String raw : value.split("(?U)\\s+")) {
                String word = raw.replaceAll("^[^\\p{L}\\p{Nd}]+|[^\\p{L}\\p{Nd}]+$", "");
                if (word.isEmpty()) continue;
                List<String> candidates =
                        word.codePoints().anyMatch(c -> c >= 128)
                                ? List.of(word)
                                : Arrays.asList(word.split("[^a-z0-9]+"));
                for (String candidate : candidates) {
                    if (candidate.isEmpty() || candidate.codePoints().allMatch(Character::isDigit)) continue;
                    if (candidate.codePoints().allMatch(c -> c < 128) && STOPWORDS.contains(candidate))
                        continue;
                    accepted.add(candidate);
                    if (accepted.size() >= 20) return List.copyOf(accepted);
                }
            }
        }
        return List.copyOf(accepted);
    }
}
