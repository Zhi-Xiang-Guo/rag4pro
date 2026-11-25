// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.*;

/**
 * Elasticsearch 存储适配层：组装查询 DSL、发送 HTTP 请求、读取命中字段并维护索引。
 *
 * <p>对照 internal/engine/elasticsearch/chunk.go，保留原 mapping、query_string、kNN filter 和二次 KNN 评分。
 * 检索重试、最终评分和分页由 StandardRetrieval 决定；阅读这里时重点观察实际发送的 body。
 */
public final class ElasticsearchStore {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final TypeReference<LinkedHashMap<String, Object>> MAP = new TypeReference<>() {
    };
    private final HttpClient client =
            HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    private final URI base;
    private final String index;

    public ElasticsearchStore(URI base, String index) {
        if (!index.matches("[a-z0-9_-]+")) throw new IllegalArgumentException("Invalid index name");
        this.base = URI.create(base.toString().replaceAll("/+$", "") + "/");
        this.index = index;
    }

    public void initialize() throws Exception {
        var response = request("HEAD", index, null);
        if (response.statusCode() == 200) return;
        if (response.statusCode() != 404) throw failure(response);
        try (var stream = getClass().getResourceAsStream("/ragflow/es-mapping.json")) {
            if (stream == null) throw new IOException("Missing upstream ES mapping");
            exchange("PUT", index, JSON.readTree(stream));
        }
    }

    public Map<String, Object> health() throws Exception {
        JsonNode result = exchange("GET", index + "/_count", null);
        return Map.of(
                "chunkCount", result.path("count").asLong(), "engine", "Elasticsearch", "index", index);
    }

    /**
     * total 是 ES 报告的命中总数，chunks 受 limit 限制；request 返回实际 DSL，便于 trace 展示。
     */
    public record Search(List<Map<String, Object>> chunks, long total, Map<String, Object> request) {
    }

    public Search search(
            String dataset,
            List<String> documentIds,
            QueryBuilder.MatchTextExpr text,
            double[] vector,
            int k,
            int numCandidates,
            int limit,
            double minimumSimilarity,
            double vectorWeight,
            Map<String, Object> filters,
            Map<String, Double> rankFeatures)
            throws Exception {
        return search(
                dataset,
                documentIds,
                text,
                vector,
                k,
                numCandidates,
                limit,
                minimumSimilarity,
                vectorWeight,
                filters,
                rankFeatures,
                List.of("*"));
    }

    public Search search(
            String dataset,
            List<String> documentIds,
            QueryBuilder.MatchTextExpr text,
            double[] vector,
            int k,
            int numCandidates,
            int limit,
            double minimumSimilarity,
            double vectorWeight,
            Map<String, Object> filters,
            Map<String, Double> rankFeatures,
            List<String> selectedFields)
            throws Exception {
        // 文本查询和 kNN 复用相同的数据集、文档与可用性约束，避免向量路径越过过滤范围。
        Map<String, Object> bool = scope(dataset, documentIds, filters);
        boolean dense = vector != null && vector.length > 0;
        if (text != null) {
            Map<String, Object> options = text.extraOptions() == null ? Map.of() : text.extraOptions();
            double msm = options.get("minimum_should_match") instanceof Number n ? n.doubleValue() : 0;
            double boost = options.get("boost") instanceof Number n ? n.doubleValue() : 1;
            List<String> fields = text.fields();
            if (fields == null || fields.isEmpty()) {
                fields =
                        List.of(
                                "title_tks^10",
                                "title_sm_tks^5",
                                "important_kwd^30",
                                "important_tks^20",
                                "question_tks^20",
                                "content_ltks^2",
                                "content_sm_ltks");
            }
            Map<String, Object> query =
                    Map.of(
                            "query_string",
                            Map.of(
                                    "fields",
                                    fields,
                                    "type",
                                    "best_fields",
                                    "query",
                                    lowerCaseQueryText(text.matchingText()),
                                    "minimum_should_match",
                                    (long) Math.floor(msm * 100 + 0.5 + 1e-9) + "%",
                                    "boost",
                                    boost));
            bool.put("must", List.of(query));
            if (dense) bool.put("boost", 1 - vectorWeight);
        }
        Map<String, Object> body = new LinkedHashMap<>();
        int effectiveLimit = limit <= 0 ? 30 : limit;
        if (dense) {
            int effectiveK = Math.min(k <= 0 ? effectiveLimit : k, 10000);
            int candidates = Math.max(effectiveK, Math.min(numCandidates, 10000));
            Map<String, Object> snapshot = JSON.convertValue(Map.of("bool", bool), MAP);
            // 原版 kNN 没有额外 boost；先复制过滤条件，再给全文查询追加排名特征。
            body.put(
                    "knn",
                    Map.of(
                            "field",
                            "q_" + vector.length + "_vec",
                            "query_vector",
                            vector,
                            "k",
                            effectiveK,
                            "num_candidates",
                            candidates,
                            "similarity",
                            minimumSimilarity,
                            "filter",
                            snapshot));
        }
        if (text != null || !dense) {
            List<Object> should = new ArrayList<>();
            if (rankFeatures != null) {
                for (var entry : new TreeMap<>(rankFeatures).entrySet()) {
                    String field =
                            entry.getKey().equals("pagerank_fea") ? "pagerank_fea" : "tag_feas." + entry.getKey();
                    should.add(
                            Map.of(
                                    "rank_feature",
                                    Map.of("field", field, "boost", entry.getValue(), "linear", Map.of())));
                }
            }
            if (!should.isEmpty()) bool.put("should", should);
            body.put("query", Map.of("bool", bool));
        }
        if (selectedFields != null && !selectedFields.isEmpty()) {
            List<String> source = new ArrayList<>(selectedFields);
            if (text != null || dense) {
                if (!source.contains("pagerank_fea")) source.add("pagerank_fea");
                if (!source.contains("tag_feas")) source.add("tag_feas");
            }
            body.put("_source", source);
            List<String> vectors =
                    selectedFields.stream().filter(field -> field.endsWith("_vec")).toList();
            if (!vectors.isEmpty()) body.put("fields", vectors);
        } else if (text != null || dense) {
            body.put("_source", List.of("pagerank_fea", "tag_feas"));
        }
        body.put("size", effectiveLimit);
        body.put("from", 0);
        body.put("track_total_hits", true);
        JsonNode result = exchange("POST", index + "/_search?allow_partial_search_results=false", body);
        JsonNode total = result.path("hits").path("total");
        List<Map<String, Object>> chunks = hits(result);
        if (text != null || dense) {
            chunks.sort(
                    Comparator.comparingDouble(
                                    (Map<String, Object> chunk) -> ((Number) chunk.get("_score")).doubleValue())
                            .reversed());
            if (chunks.size() > effectiveLimit)
                chunks = new ArrayList<>(chunks.subList(0, effectiveLimit));
        }
        return new Search(
                chunks, total.isNumber() ? total.asLong() : total.path("value").asLong(), body);
    }

    /**
     * 合并调用方的 chunk 条件与文档范围；available_int 默认排除只用于回填的隐藏父块。
     */
    private Map<String, Object> scope(
            String dataset, List<String> documents, Map<String, Object> filter) {
        Map<String, Object> conditions = new LinkedHashMap<>(filter == null ? Map.of() : filter);
        conditions.putIfAbsent("available_int", 1);
        Map<String, Object> result = conditionBool(conditions);
        List<Object> clauses = new ArrayList<>();
        clauses.add(Map.of("terms", Map.of("kb_id", List.of(dataset))));
        if (documents != null && !documents.isEmpty())
            clauses.add(Map.of("terms", Map.of("doc_id", documents)));
        if (result.get("filter") instanceof List<?> existing) clauses.addAll(existing);
        result.put("filter", clauses);
        return result;
    }

    /**
     * 将 chunk 条件转换为 bool/filter；对应原 buildBoolQueryFromConditionRaw。
     */
    private Map<String, Object> conditionBool(Map<String, Object> conditions) {
        List<Object> filters = new ArrayList<>();
        List<Object> exclusions = new ArrayList<>();
        for (var entry : conditions.entrySet()) {
            String field = entry.getKey();
            Object value = entry.getValue();
            switch (field) {
                case "and", "or" -> {
                    List<Object> children = new ArrayList<>();
                    if (value instanceof Collection<?> values) {
                        for (Object child : values) {
                            if (child instanceof Map<?, ?> map) {
                                children.add(Map.of("bool", conditionBool(JSON.convertValue(map, MAP))));
                            }
                        }
                    }
                    if (!children.isEmpty())
                        filters.add(
                                Map.of(
                                        "bool",
                                        field.equals("and")
                                                ? Map.of("filter", children)
                                                : Map.of("should", children, "minimum_should_match", 1)));
                }
                case "available_int" -> {
                    if (value instanceof Number number) {
                        Map<String, Object> range = Map.of("range", Map.of(field, Map.of("lt", 1)));
                        filters.add(
                                number.doubleValue() == 0
                                        ? range
                                        : Map.of("bool", Map.of("must_not", List.of(range))));
                    }
                }
                case "must_not" -> {
                    if (value instanceof Map<?, ?> map && !map.isEmpty()) {
                        Map<String, Object> excluded = JSON.convertValue(map, MAP);
                        exclusions.add(
                                excluded.size() == 1 && excluded.containsKey("exists")
                                        ? Map.of("exists", Map.of("field", excluded.get("exists")))
                                        : Map.of("bool", conditionBool(excluded)));
                    }
                }
                case "exists" -> filters.add(Map.of("exists", Map.of("field", value)));
                case "id", "knowledge_graph_kwd" -> {
                    String[] alternatives =
                            field.equals("id")
                                    ? new String[]{"id", "_id"}
                                    : new String[]{"type_kwd", "knowledge_graph_kwd"};
                    List<Object> clauses = new ArrayList<>();
                    for (String alternative : alternatives) {
                        if (value instanceof Collection<?> values && !values.isEmpty()) {
                            clauses.add(Map.of("terms", Map.of(alternative, values)));
                        } else if (value instanceof String string && !string.isEmpty()) {
                            clauses.add(Map.of("term", Map.of(alternative, string)));
                        } else if (field.equals("id") && value instanceof Integer number && number != 0) {
                            clauses.add(Map.of("term", Map.of(alternative, number)));
                        }
                    }
                    if (!clauses.isEmpty())
                        filters.add(Map.of("bool", Map.of("should", clauses, "minimum_should_match", 1)));
                }
                default -> {
                    if (value instanceof Collection<?> values)
                        filters.add(Map.of("terms", Map.of(field, values)));
                    else if (value instanceof String string && !string.isEmpty())
                        filters.add(Map.of("term", Map.of(field, string)));
                    else if (value instanceof Number) filters.add(Map.of("term", Map.of(field, value)));
                }
            }
        }
        Map<String, Object> result = new LinkedHashMap<>();
        if (!filters.isEmpty()) result.put("filter", filters);
        if (!exclusions.isEmpty()) result.put("must_not", exclusions);
        return result;
    }

    static String lowerCaseQueryText(String text) {
        StringBuilder result = new StringBuilder();
        boolean quoted = false;
        boolean escaped = false;
        for (int i = 0; i < text.length(); ) {
            char ch = text.charAt(i);
            if (escaped) {
                result.append(ch);
                escaped = false;
                i++;
            } else if (ch == '\\') {
                result.append(ch);
                escaped = true;
                i++;
            } else if (ch == '"') {
                result.append(ch);
                quoted = !quoted;
                i++;
            } else if (asciiLetter(ch)) {
                int end = i + 1;
                while (end < text.length() && asciiLetter(text.charAt(end))) end++;
                String word = text.substring(i, end);
                result.append(
                        !quoted && Set.of("AND", "OR", "NOT").contains(word)
                                ? word
                                : word.toLowerCase(Locale.ROOT));
                i = end;
            } else {
                result.append(ch);
                i++;
            }
        }
        return result.toString();
    }

    private static boolean asciiLetter(char value) {
        return value >= 'a' && value <= 'z' || value >= 'A' && value <= 'Z';
    }

    /**
     * 用候选 ID 限定二次 kNN 取分；返回引擎 _score，供二阶段融合使用。
     */
    public Map<String, Double> knnScores(List<Map<String, Object>> chunks, double[] query)
            throws Exception {
        if (chunks.isEmpty() || query.length == 0) return Map.of();
        Set<String> ids = new LinkedHashSet<>();
        Set<String> indices = new LinkedHashSet<>();
        for (Map<String, Object> chunk : chunks) {
            if (chunk.get("_id") instanceof String id && !id.isEmpty()) {
                if (!(chunk.get("_index") instanceof String candidateIndex) || candidateIndex.isEmpty()) {
                    throw new IllegalArgumentException("KNN candidate " + id + " has no index");
                }
                if (!candidateIndex.matches("[a-z0-9_-]+"))
                    throw new IllegalArgumentException("Invalid candidate index");
                ids.add(id);
                indices.add(candidateIndex);
            }
        }
        if (ids.isEmpty()) return Map.of();
        Map<String, Object> knn =
                Map.of(
                        "field",
                        "q_" + query.length + "_vec",
                        "query_vector",
                        query,
                        "k",
                        ids.size(),
                        "num_candidates",
                        ids.size() * 2,
                        "similarity",
                        0.0,
                        "filter",
                        Map.of("terms", Map.of("id", ids)));
        var result =
                exchange(
                        "POST",
                        String.join(",", indices) + "/_search?allow_partial_search_results=false",
                        Map.of("knn", knn, "size", ids.size(), "_source", false));
        Map<String, Double> scores = new LinkedHashMap<>();
        for (var hit : result.path("hits").path("hits")) {
            scores.put(hit.path("_id").asText(), hit.path("_score").asDouble());
        }
        return scores;
    }

    /**
     * 按 chunk ID 投影字段，并保留原版单值/多值转换规则，供后续评分按 ID 取值。
     */
    public Map<String, Map<String, Object>> getFields(
            List<Map<String, Object>> chunks, List<String> fields) {
        Map<String, Map<String, Object>> result = new LinkedHashMap<>();
        for (Map<String, Object> chunk : chunks) {
            String id = chunkId(chunk);
            if (id.isEmpty()) continue;
            Map<String, Object> projected = new LinkedHashMap<>();
            for (String field : fields) {
                Object value = field.equals("id") ? id : chunk.get(field);
                if (value == null) continue;
                if (value instanceof Map<?, ?>) {
                    try {
                        projected.put(field, JSON.writeValueAsString(value));
                    } catch (IOException failure) {
                        throw new IllegalArgumentException("Cannot encode " + field, failure);
                    }
                    continue;
                }
                if (value instanceof List<?> list
                        && list.size() == 1
                        && !(list.getFirst() instanceof List<?>)) {
                    value = list.getFirst();
                }
                if (value instanceof List<?> || field.equals("available_int") && value instanceof Number) {
                    projected.put(field, value);
                } else {
                    projected.put(field, goScalarString(value));
                }
            }
            if (!projected.isEmpty()) result.put(id, projected);
        }
        return result;
    }

    private static String goScalarString(Object value) {
        if (value == null) return "<nil>";
        if (value instanceof Double number && number == Math.rint(number))
            return String.format(Locale.ROOT, "%.0f", number);
        if (value instanceof Float number && number == Math.rint(number))
            return String.format(Locale.ROOT, "%.0f", number);
        return value.toString();
    }

    public Map<String, String> getHighlight(
            List<Map<String, Object>> chunks, List<String> keywords, String field) {
        List<String> sorted =
                keywords.stream()
                        .filter(word -> !word.isEmpty())
                        .sorted(
                                Comparator.comparingInt((String word) -> word.codePointCount(0, word.length()))
                                        .reversed())
                        .toList();
        List<String> alternatives = new ArrayList<>();
        for (String word : sorted) {
            boolean latin =
                    word.codePoints()
                            .allMatch(cp -> Character.UnicodeScript.of(cp) == Character.UnicodeScript.LATIN);
            alternatives.add(java.util.regex.Pattern.quote(word) + (latin ? "\\p{IsLatin}*" : ""));
        }
        var pattern =
                alternatives.isEmpty()
                        ? null
                        : java.util.regex.Pattern.compile(
                        String.join("|", alternatives),
                        java.util.regex.Pattern.CASE_INSENSITIVE | java.util.regex.Pattern.UNICODE_CASE);
        Map<String, String> result = new LinkedHashMap<>();
        for (Map<String, Object> chunk : chunks) {
            String id = chunkId(chunk);
            if (id.isEmpty() || !(chunk.get(field) instanceof String text)) continue;
            if (pattern != null) {
                text =
                        pattern
                                .matcher(text)
                                .replaceAll(
                                        match ->
                                                java.util.regex.Matcher.quoteReplacement("<em>" + match.group() + "</em>"));
            }
            result.put(id, text);
        }
        return result;
    }

    private static String chunkId(Map<String, Object> chunk) {
        if (chunk.get("id") instanceof String id && !id.isEmpty()) return id;
        return chunk.get("_id") instanceof String id ? id : "";
    }

    public synchronized List<Map<String, Object>> chunks(String dataset, String document)
            throws Exception {
        Map<String, Object> query =
                Map.of("bool", scope(dataset, document == null ? List.of() : List.of(document), Map.of()));
        return scan(query);
    }

    public synchronized List<Map<String, Object>> allDocumentChunks(String dataset, String document)
            throws Exception {
        List<Object> filters =
                List.of(
                        Map.of("term", Map.of("kb_id", dataset)), Map.of("term", Map.of("doc_id", document)));
        return scan(Map.of("bool", Map.of("filter", filters)));
    }

    private List<Map<String, Object>> scan(Map<String, Object> query) throws Exception {
        List<Map<String, Object>> all = new ArrayList<>();
        Map<String, Object> request = new LinkedHashMap<>();
        request.put("query", query);
        request.put("size", 1000);
        request.put("sort", List.of(Map.of("id", "asc")));
        while (true) {
            var response = exchange("POST", index + "/_search", request);
            var batch = hits(response);
            all.addAll(batch);
            var raw = response.path("hits").path("hits");
            if (batch.size() < 1000) break;
            request.put(
                    "search_after", JSON.convertValue(raw.get(raw.size() - 1).path("sort"), List.class));
        }
        return all;
    }

    public synchronized void replace(String dataset, String docId, List<Map<String, Object>> chunks)
            throws Exception {
        List<Map<String, Object>> old = allDocumentChunks(dataset, docId);
        try {
            bulk(chunks, dataset);
            Set<String> newIds = new HashSet<>();
            for (var chunk : chunks) newIds.add(chunk.get("id").toString());
            List<String> stale =
                    old.stream().map(c -> c.get("id").toString()).filter(id -> !newIds.contains(id)).toList();
            if (!stale.isEmpty()) deleteIds(stale);
        } catch (Exception failure) {
            try {
                deleteDocument(dataset, docId);
                bulk(old, dataset);
            } catch (Exception rollback) {
                failure.addSuppressed(rollback);
            }
            throw failure;
        }
    }

    /** 仅新增批次；失败时清理本批 ID，部分 bulk 成功也不会成为可检索文档。 */
    public synchronized void createBatch(String dataset, List<Map<String, Object>> chunks) throws Exception {
        try {
            bulk(chunks, dataset);
        } catch (Exception failure) {
            try { removeBatch(chunks); } catch (Exception rollback) { failure.addSuppressed(rollback); }
            throw failure;
        }
    }

    public synchronized void removeBatch(List<Map<String, Object>> chunks) throws Exception {
        deleteIds(chunks.stream().map(c -> c.get("id").toString()).toList());
    }

    private void bulk(List<Map<String, Object>> chunks, String dataset) throws Exception {
        StringBuilder body = new StringBuilder();
        int count = 0;
        int bytes = 0;
        for (var chunk : chunks) {
            if (!(chunk.get("id") instanceof String id)
                    || id.isEmpty()
                    || !(chunk.get("doc_id") instanceof String docId)
                    || docId.isEmpty()) continue;
            Map<String, Object> source = new LinkedHashMap<>(chunk);
            source.remove("_id");
            source.remove("_index");
            source.remove("_score");
            source.remove("highlight");
            source.put("kb_id", dataset);
            source.putIfAbsent("available_int", 1);
            if (source.get("tag_feas") instanceof Map<?, ?> features)
                source.put("tag_feas", orderedTagFeatures(features));
            String pair =
                    JSON.writeValueAsString(Map.of("index", Map.of("_index", index, "_id", id)))
                            + "\n"
                            + JSON.writeValueAsString(source)
                            + "\n";
            body.append(pair);
            count++;
            bytes += pair.getBytes(StandardCharsets.UTF_8).length;
            if (count >= 500 || bytes >= 20 * 1024 * 1024) {
                flushBulk(body.toString());
                body.setLength(0);
                count = 0;
                bytes = 0;
            }
        }
        if (!body.isEmpty()) flushBulk(body.toString());
    }

    private void flushBulk(String body) throws Exception {
        var response = request("POST", index + "/_bulk?refresh=wait_for", body);
        if (response.statusCode() >= 300) throw failure(response);
        JsonNode result = JSON.readTree(response.body());
        if (result.path("errors").asBoolean())
            throw new IOException("Elasticsearch bulk item failed: " + result.path("items"));
    }

    private static Map<String, Integer> orderedTagFeatures(Map<?, ?> values) {
        Map<String, Integer> rounded = new HashMap<>();
        values.forEach(
                (key, value) -> {
                    if (value instanceof Number score) {
                        double number = score.doubleValue();
                        rounded.put(key.toString(), (int) (number < 0 ? number - 0.5 : number + 0.5));
                    }
                });
        Map<String, Integer> ordered = new LinkedHashMap<>();
        rounded.entrySet().stream()
                .sorted(
                        Map.Entry.<String, Integer>comparingByValue()
                                .reversed()
                                .thenComparing(Map.Entry.comparingByKey()))
                .forEach(entry -> ordered.put(entry.getKey(), entry.getValue()));
        return ordered;
    }

    public synchronized void deleteDocument(String dataset, String document) throws Exception {
        exchange(
                "POST",
                index + "/_delete_by_query?refresh=true&conflicts=proceed",
                Map.of(
                        "query",
                        Map.of(
                                "bool",
                                Map.of(
                                        "filter",
                                        List.of(
                                                Map.of("term", Map.of("kb_id", dataset)),
                                                Map.of("term", Map.of("doc_id", document)))))));
    }

    private void deleteIds(List<String> ids) throws Exception {
        exchange(
                "POST",
                index + "/_delete_by_query?refresh=true&conflicts=proceed",
                Map.of("query", Map.of("ids", Map.of("values", ids))));
    }

    public Map<String, Object> getParent(String parentId, String documentId, String datasetId)
            throws Exception {
        Map<String, Object> filter =
                conditionBool(Map.of("id", parentId, "doc_id", documentId, "kb_id", datasetId));
        JsonNode response =
                exchange(
                        "POST",
                        index + "/_search?allow_partial_search_results=false",
                        Map.of("query", Map.of("bool", filter), "size", 1, "_source", List.of("*")));
        List<Map<String, Object>> rows = hits(response);
        return rows.isEmpty() ? null : rows.getFirst();
    }

    public Map<String, Object> get(String id) throws Exception {
        var response =
                request(
                        "GET", index + "/_doc/" + java.net.URLEncoder.encode(id, StandardCharsets.UTF_8), null);
        if (response.statusCode() == 404) return null;
        if (response.statusCode() >= 300) throw failure(response);
        Map<String, Object> result =
                JSON.convertValue(JSON.readTree(response.body()).path("_source"), MAP);
        result.put("id", id);
        return result;
    }

    private List<Map<String, Object>> hits(JsonNode result) {
        List<Map<String, Object>> chunks = new ArrayList<>();
        for (var hit : result.path("hits").path("hits")) {
            Map<String, Object> chunk =
                    hit.path("_source").isObject()
                            ? JSON.convertValue(hit.path("_source"), MAP)
                            : new LinkedHashMap<>();
            chunk.put("id", hit.path("_id").asText());
            chunk.put("_id", hit.path("_id").asText());
            chunk.put("_index", hit.path("_index").asText());
            chunk.put("_score", hit.path("_score").asDouble());
            if (hit.path("highlight").isObject())
                chunk.put("highlight", JSON.convertValue(hit.path("highlight"), MAP));
            chunks.add(chunk);
        }
        return chunks;
    }

    private JsonNode exchange(String method, String path, Object body) throws Exception {
        var response = request(method, path, body == null ? null : JSON.writeValueAsString(body));
        if (response.statusCode() >= 300) throw failure(response);
        return JSON.readTree(response.body());
    }

    private HttpResponse<String> request(String method, String path, String body) throws Exception {
        var builder =
                HttpRequest.newBuilder(base.resolve(path))
                        .timeout(Duration.ofSeconds(90))
                        .header(
                                "Content-Type",
                                path.contains("/_bulk") ? "application/x-ndjson" : "application/json");
        builder.method(
                method,
                body == null
                        ? HttpRequest.BodyPublishers.noBody()
                        : HttpRequest.BodyPublishers.ofString(body));
        return client.send(builder.build(), HttpResponse.BodyHandlers.ofString());
    }

    private IOException failure(HttpResponse<String> response) {
        return new IOException("Elasticsearch HTTP " + response.statusCode() + ": " + response.body());
    }
}
