// Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations.
package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;

/**
 * 逐分支移植 internal/service/nlp/reranker.go，保留字段名、计算次序及退化行为。
 */
public final class Reranker {
    public static final double DEFAULT_VECTOR_WEIGHT = 0.3;
    public static final double DEFAULT_TOKEN_WEIGHT = 1 - DEFAULT_VECTOR_WEIGHT;
    public static final double DEFAULT_VECTOR_ONLY_WEIGHT = 1.0;
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Pattern SPACE = Pattern.compile("\\s+", Pattern.UNICODE_CHARACTER_CLASS);
    private static final Comparator<String> GO_STRING_ORDER =
            (left, right) ->
                    Arrays.compareUnsigned(
                            left.getBytes(StandardCharsets.UTF_8), right.getBytes(StandardCharsets.UTF_8));

    private Reranker() {
    }

    public record TermWeight(String term, double weight) {
    }

    /**
     * 三个数组按候选顺序对齐：sim 为最终分，tsim 为词法分，vsim 为向量或模型分。
     *
     * <p>sim 可能额外叠加 PageRank/标签特征，不能当成概率；vsim 的含义需结合实际评分分支查看。
     */
    public record Result(double[] sim, double[] tsim, double[] vsim) {
    }

    public record ModelScore(int index, double relevanceScore) {
    }

    /**
     * question 对接 Question(query,"qa",0.6,language)，weights 对接 Weights(tokens,false,language)。
     */
    public interface TokenSource {
        List<String> question(String query, String language);

        List<TermWeight> weights(List<String> tokens, String language);
    }

    @FunctionalInterface
    public interface Model {
        List<ModelScore> rerank(String query, List<String> documents) throws Exception;
    }

    /**
     * 原 RerankStandard。缺失向量按零向量处理，全部 cosine 为零时直接使用词法分数。
     */
    public static Result rerankStandard(
            List<Map<String, Object>> chunks,
            List<String> keywords,
            double[] questionVector,
            String query,
            double tkWeight,
            double vtWeight,
            String cfield,
            TokenSource source,
            Map<String, Double> rankFeature,
            String language) {
        if (chunks == null || chunks.isEmpty()) return empty();
        if (keywords == null) keywords = List.of();
        if (source != null && keywords.isEmpty()) keywords = source.question(query, language);
        if (questionVector == null) questionVector = new double[0];
        List<double[]> vectors = new ArrayList<>();
        List<List<String>> tokens = new ArrayList<>();
        String vectorColumn = "q_" + questionVector.length + "_vec";
        for (Map<String, Object> chunk : chunks) {
            vectors.add(extractVector(chunk, vectorColumn, new double[questionVector.length]));
            tokens.add(buildRerankTokens(chunk, cfield, 2, 5, 6));
        }
        Result result =
                hybridSimilarity(
                        questionVector, vectors, keywords, tokens, tkWeight, vtWeight, source, language);
        applyRankFeatureScores(chunks, result.sim(), rankFeature);
        return result;
    }

    /**
     * 原 RerankWithKNN。KNN 分数已由后端提取，按 ID 取值并保留未命中的零分。
     */
    public static Result rerankWithKNN(
            List<Map<String, Object>> chunks,
            List<String> ids,
            Map<String, Map<String, Object>> field,
            Map<String, Double> knnScores,
            String query,
            double tkWeight,
            double vtWeight,
            String cfield,
            TokenSource source,
            Map<String, Double> rankFeature,
            String language) {
        if (ids == null || ids.isEmpty()) return empty();
        List<List<String>> tokens = new ArrayList<>();
        double[] vectorScores = new double[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            String id = ids.get(i);
            Map<String, Object> chunk = field == null ? null : field.get(id);
            if (chunk != null && chunk.get("important_kwd") instanceof String keyword) {
                chunk.put("important_kwd", new String[]{keyword});
            }
            if (chunk == null && chunks != null && i < chunks.size()) chunk = chunks.get(i);
            tokens.add(buildRerankTokens(chunk, cfield, 2, 5, 6));
            vectorScores[i] = knnScores == null ? 0 : knnScores.getOrDefault(id, 0.0);
        }
        List<String> keywords = source == null ? List.of() : source.question(query, language);
        double[] tokenScores = tokenSimilarity(keywords, tokens, source, language);
        double[] scores = blend(tokenScores, vectorScores, tkWeight, vtWeight);
        applyRankFeatureScoresForIDs(ids, field, scores, rankFeature);
        return new Result(scores, tokenScores, vectorScores);
    }

    /**
     * 原 RerankByModel。保留自然文本，按响应 index 归位，归一化后融合并叠加排名特征。
     */
    public static Result rerankByModel(
            Model model,
            List<Map<String, Object>> chunks,
            List<String> ids,
            Map<String, Map<String, Object>> field,
            String query,
            double tkWeight,
            double vtWeight,
            String cfield,
            TokenSource source,
            Map<String, Double> rankFeature,
            String language)
            throws Exception {
        if (chunks == null || chunks.isEmpty()) return empty();
        if (ids == null) ids = List.of();
        List<List<String>> tokens = new ArrayList<>();
        List<String> documents = new ArrayList<>();
        for (int i = 0; i < ids.size(); i++) {
            Map<String, Object> chunk = field == null ? null : field.get(ids.get(i));
            if (chunk == null) {
                if (i >= chunks.size()) continue;
                chunk = chunks.get(i);
            }
            List<String> row = buildRerankTokens(chunk, cfield, 1, 1, 1);
            tokens.add(row);
            Object natural = chunk.get("content_with_weight");
            String text = natural instanceof String value ? value : "";
            if (text.isEmpty()) text = removeRedundantSpaces(String.join(" ", row));
            documents.add(text);
        }
        List<String> keywords = source == null ? List.of() : source.question(query, language);
        double[] tokenScores = tokenSimilarity(keywords, tokens, source, language);
        double[] modelScores = new double[tokens.size()];
        for (ModelScore score : model.rerank(query, documents)) {
            if (score.index() >= 0 && score.index() < modelScores.length)
                modelScores[score.index()] = score.relevanceScore();
        }
        normalizeRerankScores(modelScores);
        double[] scores = blend(tokenScores, modelScores, tkWeight, vtWeight);
        applyRankFeatureScoresForIDs(ids, field, scores, rankFeature);
        return new Result(scores, tokenScores, modelScores);
    }

    public static Result hybridSimilarity(
            double[] queryVector,
            List<double[]> vectors,
            List<String> queryTokens,
            List<List<String>> documentTokens,
            double tkWeight,
            double vtWeight,
            TokenSource source,
            String language) {
        double[] vectorScores = new double[vectors.size()];
        boolean allZero = true;
        for (int i = 0; i < vectors.size(); i++) {
            vectorScores[i] = cosineSimilarity(queryVector, vectors.get(i));
            if (vectorScores[i] != 0) allZero = false;
        }
        double[] tokenScores = tokenSimilarity(queryTokens, documentTokens, source, language);
        double[] scores =
                allZero ? tokenScores.clone() : blend(tokenScores, vectorScores, tkWeight, vtWeight);
        return new Result(scores, tokenScores, vectorScores);
    }

    // 普通融合分 = 词法权重 × 词法覆盖分 + 向量权重 × 向量/模型分；排名特征在调用方叠加。
    private static double[] blend(
            double[] tokens, double[] vectors, double tkWeight, double vtWeight) {
        double[] scores = new double[tokens.length];
        for (int i = 0; i < scores.length; i++)
            scores[i] = tkWeight * tokens[i] + vtWeight * vectors[i];
        return scores;
    }

    /**
     * 按查询侧权重计算词及相邻组合的覆盖程度；这不是 ES 首次召回的全文检索分数。
     */
    public static double[] tokenSimilarity(
            List<String> queryTokens,
            List<List<String>> documentTokens,
            TokenSource source,
            String language) {
        if (documentTokens == null) return new double[0];
        Map<String, Double> query = tokensToDict(queryTokens, source, language);
        double[] scores = new double[documentTokens.size()];
        for (int i = 0; i < scores.length; i++) {
            scores[i] =
                    tokenDictSimilarity(
                            query,
                            tokensToDict(documentTokens.get(i), source, language),
                            new ArrayList<>(query.keySet()));
        }
        return scores;
    }

    /**
     * 原 tokensToDict。单词 0.4，相邻拼接词 0.6，保留首次插入顺序和重复词累加。
     */
    static Map<String, Double> tokensToDict(
            List<String> tokens, TokenSource source, String language) {
        Map<String, Double> weights = new LinkedHashMap<>();
        if (source == null) return weights;
        List<TermWeight> terms = source.weights(tokens, language);
        for (int i = 0; i < terms.size(); i++) {
            TermWeight term = terms.get(i);
            weights.merge(term.term(), term.weight() * 0.4, Double::sum);
            if (i + 1 < terms.size()) {
                TermWeight next = terms.get(i + 1);
                weights.merge(
                        term.term() + next.term(), Math.max(term.weight(), next.weight()) * 0.6, Double::sum);
            }
        }
        return weights;
    }

    /**
     * 原 tokenDictSimilarity。分母为查询权重 L1 和，仅检查文档键存在性，保留 1e-9 平滑。
     */
    static double tokenDictSimilarity(
            Map<String, Double> query, Map<String, Double> document, List<String> keyOrder) {
        if (query.isEmpty() || document.isEmpty()) return 0;
        if (keyOrder == null || keyOrder.isEmpty())
            keyOrder = query.keySet().stream().sorted(GO_STRING_ORDER).toList();
        double matching = 1e-9;
        double total = 1e-9;
        for (String key : keyOrder) if (document.containsKey(key)) matching += query.get(key);
        for (String key : keyOrder) total += query.get(key);
        return matching / total;
    }

    public static double[] normalizeRerankScores(double[] scores) {
        if (scores == null || scores == null || scores.length == 0) return scores;
        double min = scores[0], max = scores[0];
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] < min) min = scores[i];
            if (scores[i] > max) max = scores[i];
        }
        if (min >= 0 && max <= 1) return scores;
        double span = max - min;
        if (span < 1e-3) {
            for (int i = 0; i < scores.length; i++) {
                if (scores[i] < 0) scores[i] = 0;
                else if (scores[i] > 1) scores[i] = 1;
            }
            return scores;
        }
        double inverseSpan = 1 / span;
        for (int i = 0; i < scores.length; i++) scores[i] = (scores[i] - min) * inverseSpan;
        return scores;
    }

    /**
     * 把正文、标题、关键词、预置问题组成重排词序列；重复加入用于表达字段权重。
     */
    static List<String> buildRerankTokens(
            Map<String, Object> chunk,
            String cfield,
            int titleRepeat,
            int importantRepeat,
            int questionRepeat) {
        if (chunk == null) return new ArrayList<>();
        List<String> tokens = new ArrayList<>(new LinkedHashSet<>(split(chunk.get(cfield))));
        for (int i = 0; i < titleRepeat; i++) tokens.addAll(split(chunk.get("title_tks")));
        List<String> important = new ArrayList<>();
        Object value = chunk.get("important_kwd");
        if (value instanceof String text) important.add(text);
        else if (value instanceof List<?> values) {
            for (Object item : values) if (item instanceof String text) important.add(text);
        } else if (value instanceof String[] values) important.addAll(Arrays.asList(values));
        for (int i = 0; i < importantRepeat; i++) tokens.addAll(important);
        for (int i = 0; i < questionRepeat; i++) tokens.addAll(split(chunk.get("question_tks")));
        return tokens;
    }

    static List<String> split(Object value) {
        if (!(value instanceof String text)) return List.of();
        return SPACE.splitAsStream(text).filter(token -> !token.isEmpty()).toList();
    }

    static String removeRedundantSpaces(String text) {
        text = text.replaceAll("([^\\p{L}\\p{N}_.,)>]) +([^ ])", "$1$2");
        return text.replaceAll("([^ ]) +([^\\p{L}\\p{N}_.,(<])", "$1$2");
    }

    static double[] extractVector(Map<String, Object> chunk, String column, double[] zeroVector) {
        Object value = chunk.get(column);
        if (value instanceof double[] vector) return vector;
        if (value instanceof List<?> values) {
            double[] vector = new double[values.size()];
            // 原 []interface{} 分支断言 float64，错误类型保留失败行为。
            for (int i = 0; i < vector.length; i++) vector[i] = (Double) values.get(i);
            return vector;
        }
        return zeroVector;
    }

    public static double cosineSimilarity(double[] a, double[] b) {
        if (a == null) a = new double[0];
        if (b == null) b = new double[0];
        if (a.length != b.length) return 0;
        double[] dot = new double[a.length], normA = new double[a.length], normB = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            dot[i] = a[i] * b[i];
            normA[i] = a[i] * a[i];
            normB[i] = b[i] * b[i];
        }
        double product = pairwiseSum(dot), squaredA = pairwiseSum(normA), squaredB = pairwiseSum(normB);
        if (squaredA == 0 || squaredB == 0) return 0;
        return product / (Math.sqrt(squaredA) * Math.sqrt(squaredB));
    }

    /**
     * 原 common.PairwiseSum，长度小于16顺序累加，大数组逐层两两归约。
     */
    static double pairwiseSum(double[] values) {
        if (values.length < 16) {
            double sum = 0;
            for (double value : values) sum += value;
            return sum;
        }
        int length = values.length;
        while (length > 1) {
            int pairs = length / 2;
            for (int i = 0; i < pairs; i++) values[i] = values[i * 2] + values[i * 2 + 1];
            if (length % 2 != 0) values[pairs++] = values[length - 1];
            length = pairs;
        }
        return values[0];
    }

    /**
     * 原 applyRankFeatureScores，tag cosine * 10 + pagerank，只更新重叠前缀。
     */
    public static double[] applyRankFeatureScores(
            List<Map<String, Object>> chunks, double[] scores, Map<String, Double> rankFeature) {
        if (chunks == null || chunks.isEmpty() || scores.length == 0) return scores;
        if (chunks.size() != scores.length)
            System.getLogger(Reranker.class.getName())
                    .log(System.Logger.Level.WARNING, "rank feature: sim/chunks length mismatch");
        Map<String, Double> query = rankFeature == null ? Map.of() : rankFeature;
        double[] squares =
                query.entrySet().stream()
                        .sorted(Map.Entry.comparingByKey(GO_STRING_ORDER))
                        .filter(entry -> !entry.getKey().equals("pagerank_fea"))
                        .mapToDouble(entry -> entry.getValue() * entry.getValue())
                        .toArray();
        double queryNorm = Math.sqrt(pairwiseSum(squares));
        for (int i = 0; i < Math.min(chunks.size(), scores.length); i++) {
            Map<String, Object> chunk = chunks.get(i);
            if (chunk == null) continue;
            Double pagerank = toFloat64(chunk.get("pagerank_fea"));
            double tagScore = 0;
            if (queryNorm != 0) {
                Map<String, Double> tags = extractTagFeasMap(chunk.get("tag_feas"));
                double numerator = 0, denominator = 0;
                for (String key : tags.keySet().stream().sorted(GO_STRING_ORDER).toList()) {
                    double weight = tags.get(key);
                    if (query.containsKey(key)) numerator += query.get(key) * weight;
                    denominator += weight * weight;
                }
                if (denominator != 0) tagScore = numerator / Math.sqrt(denominator) / queryNorm;
            }
            scores[i] += tagScore * 10 + (pagerank == null ? 0 : pagerank);
        }
        return scores;
    }

    public static double[] applyRankFeatureScoresForIDs(
            List<String> ids,
            Map<String, Map<String, Object>> field,
            double[] scores,
            Map<String, Double> rankFeature) {
        List<Map<String, Object>> chunks = new ArrayList<>();
        if (ids != null) for (String id : ids) chunks.add(field == null ? null : field.get(id));
        return applyRankFeatureScores(chunks, scores, rankFeature);
    }

    static Double toFloat64(Object value) {
        if (value instanceof Double
                || value instanceof Float
                || value instanceof Integer
                || value instanceof Long) return ((Number) value).doubleValue();
        if (value instanceof String text) {
            try {
                return Double.parseDouble(text.strip());
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    static Map<String, Double> extractTagFeasMap(Object value) {
        Map<String, Double> result = new LinkedHashMap<>();
        if (value == null) return result;
        try {
            Object parsed = value;
            if (value instanceof String text) parsed = JSON.readValue(text, Object.class);
            else if (!(value instanceof Map<?, ?>))
                parsed = JSON.readValue(JSON.writeValueAsString(value), Object.class);
            if (parsed instanceof Map<?, ?> map) {
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    Double number = toFloat64(entry.getValue());
                    if (entry.getKey() instanceof String key && number != null) result.put(key, number);
                }
            }
        } catch (Exception ignored) {
            /* 原 JSON 解析失败返回空字典。 */
        }
        return result;
    }

    public static int[] argsortDescending(double[] values) {
        Integer[] indices = new Integer[values.length];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> values[a] > values[b] ? -1 : values[b] > values[a] ? 1 : 0);
        return Arrays.stream(indices).mapToInt(Integer::intValue).toArray();
    }

    private static Result empty() {
        return new Result(new double[0], new double[0], new double[0]);
    }
}
