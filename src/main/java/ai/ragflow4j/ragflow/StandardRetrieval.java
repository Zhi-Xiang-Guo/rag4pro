// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
// http://www.apache.org/licenses/LICENSE-2.0
package ai.ragflow4j.ragflow;

import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * 检索流程控制器，对照 internal/service/nlp/retrieval.go 的 Elasticsearch 标准路径。
 *
 * <p>阅读主线：retrieve → search → searchEngine → scoreSearchResult → responseChunk。 QueryBuilder
 * 负责词法表达式，ElasticsearchStore 负责引擎请求，Reranker 负责二阶段评分。
 */
public final class StandardRetrieval {
    // 召回后需要读取的字段；查询向量维度确定后，searchEngine 还会追加相应的向量列。
    public static final List<String> SOURCE_FIELDS =
            List.of(
                    "docnm_kwd",
                    "content_ltks",
                    "kb_id",
                    "img_id",
                    "title_tks",
                    "important_kwd",
                    "position_int",
                    "doc_id",
                    "chunk_order_int",
                    "page_num_int",
                    "top_int",
                    "create_timestamp_flt",
                    "type_kwd",
                    "knowledge_graph_kwd",
                    "content_with_weight",
                    "doc_type_kwd",
                    "mom_id",
                    "row_id()",
                    "compile_kwd",
                    "scope_kwd",
                    "question_tks",
                    "question_kwd",
                    "pagerank_fea",
                    "tag_feas",
                    "_score");
    private final ElasticsearchStore store;
    private final QueryBuilder queryBuilder;
    private final Embedding embedding;

    @FunctionalInterface
    public interface Embedding {
        double[] query(String text) throws Exception;
    }

    /**
     * null 包装类型遵循 Go 指针参数默认值，显式零值保留其含义。
     */
    public static final class Request {
        public String question = "";
        public String datasetId;
        public List<String> docIds = List.of();
        public int page = 1;
        public int pageSize = 1;
        public Integer rerankCandidatesCount;
        public Integer knnTopK;
        public Integer knnNumCandidates;
        public Double similarityThreshold;
        public Double vectorSimilarityWeight;
        public Map<String, Double> rankFeature;
        public Reranker.Model rerankModel;
        public Boolean aggs;
        public Boolean highlight;
        public Boolean allowDenseFallback;
        public boolean vectorOnly;
        /** standard 保留原版；其余为显式的独立对照/改进，不改变原版评分含义。 */
        public String strategy = "standard";
        public Map<String, Object> filter = Map.of();
        public String language = "";

        /**
         * 由业务文档目录提供存在的源文档ID，null 表示调用方未配置目录校验。
         */
        public Set<String> existingDocumentIds;
    }

    /**
     * chunks 是当前页；total 是候选池中通过阈值的块数；docAggs 跨当前候选池聚合。
     */
    public record Result(
            List<Map<String, Object>> chunks,
            List<Map<String, Object>> docAggs,
            long total,
            List<Map<String, Object>> trace) {
    }

    // ids 确定候选顺序，fields 按 ID 存字段；评分数组的第 i 项始终对应 ids 的第 i 项。
    private record SearchResult(
            List<Map<String, Object>> chunks,
            List<String> ids,
            Map<String, Map<String, Object>> fields,
            double[] vector,
            Map<String, String> highlights,
            long total) {
    }

    private record ParentKey(String parentId, String documentId, String datasetId) {
    }

    public StandardRetrieval(
            ElasticsearchStore store, QueryBuilder queryBuilder, Embedding embedding) {
        this.store = store;
        this.queryBuilder = queryBuilder;
        this.embedding = embedding;
    }

    /**
     * 原 Retrieval，召回窗口、评分、阈值、稳定排序、分页和文档聚合的执行顺序保持一致。
     */
    public Result retrieve(Request request) throws Exception {
        List<Map<String, Object>> trace = new ArrayList<>();
        if (request.question.isEmpty()) return empty(trace);
        // 1. 补默认值并验证候选池足以覆盖所请求的页。
        defaultsAndValidate(request);
        if (!request.strategy.equals("standard")) return independent(request, trace);
        // 2. 从 ES 召回候选，再依据文档目录剔除源文档已不存在的块。
        SearchResult search = prune(search(request, trace), request.existingDocumentIds);
        if (search.total() == 0) return empty(trace);
        // 3. 对候选重新评分。ES 首次召回的 _score 不直接等于最终 similarity。
        Reranker.Result scores = scoreSearchResult(request, search, trace);
        if (scores.sim().length == 0) return empty(trace);
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < scores.sim().length; i++) indices.add(i);
        // 4. 仅排序下标，保留 ID、字段和三组评分之间的对应关系；同分时保持候选原顺序。
        indices.sort(
                (left, right) ->
                        scores.sim()[left] > scores.sim()[right]
                                ? -1
                                : scores.sim()[right] > scores.sim()[left] ? 1 : 0);
        // 5. 先过滤再分页；纯词法和纯向量分支沿用原版的零阈值规则。
        double threshold = request.similarityThreshold;
        if (request.vectorSimilarityWeight <= 0 || request.vectorOnly) threshold = 0;
        List<Integer> valid = new ArrayList<>();
        for (int index : indices) if (scores.sim()[index] >= threshold) valid.add(index);
        if (valid.isEmpty()) return empty(trace);
        int begin = (request.page - 1) * request.pageSize;
        int end = Math.min(begin + request.pageSize, valid.size());
        List<Map<String, Object>> chunks = new ArrayList<>();
        for (int page = begin; page < end; page++) {
            int index = valid.get(page);
            if (index >= search.ids().size()) continue;
            String id = search.ids().get(index);
            Map<String, Object> fields = search.fields().get(id);
            if (fields != null) chunks.add(responseChunk(id, fields, index, scores, search));
        }
        // 文档聚合使用全部过阈值的候选，因此数量可能大于当前页的 chunks 数量。
        List<Map<String, Object>> aggregations = request.aggs ? aggregate(valid, search) : List.of();
        trace.add(
                Map.of(
                        "stage",
                        "pagination",
                        "candidateCount",
                        search.ids().size(),
                        "threshold",
                        threshold,
                        "validCount",
                        valid.size(),
                        "page",
                        request.page,
                        "pageSize",
                        request.pageSize));
        return new Result(chunks, aggregations, valid.size(), trace);
    }

    /** 两路独立召回避免词法条件预先过滤 dense 结果，再以 rank 融合消除分数量纲差异。 */
    private Result independent(Request request, List<Map<String, Object>> trace) throws Exception {
        if (!Set.of("lexical", "dense", "rrf").contains(request.strategy))
            throw new IllegalArgumentException("unknown retrieval strategy");
        if (request.rerankModel != null) throw new IllegalArgumentException("independent strategies do not use model reranking");
        LinkedHashMap<String, Map<String, Object>> fields = new LinkedHashMap<>();
        List<List<String>> rankings = new ArrayList<>();
        for (String branch : request.strategy.equals("rrf") ? List.of("lexical", "dense") : List.of(request.strategy)) {
            long started = System.nanoTime();
            QueryBuilder.MatchTextExpr expression = branch.equals("lexical")
                    ? queryBuilder.question(request.question, "qa", 0.3, request.language).expression() : null;
            // 空词法表达式应返回空列表，不能退化成 match-all。
            if (branch.equals("lexical") && expression == null) { rankings.add(List.of()); continue; }
            double[] vector = branch.equals("dense") ? embedding.query(request.question) : new double[0];
            // RRF 将总候选预算分给两路，避免用双倍候选数制造表面收益。
            int branchLimit = request.strategy.equals("rrf")
                    ? (branch.equals("lexical") ? Math.max(1, request.rerankCandidatesCount / 2)
                            : Math.max(1, (request.rerankCandidatesCount + 1) / 2))
                    : request.rerankCandidatesCount;
            var search = store.search(request.datasetId, request.docIds, expression, vector,
                    request.knnTopK, request.knnNumCandidates, branchLimit, 0, 1,
                    request.filter, request.rankFeature, SOURCE_FIELDS);
            List<String> ids = new ArrayList<>();
            for (var chunk : search.chunks()) {
                if (request.existingDocumentIds != null
                        && !request.existingDocumentIds.contains(firstString(chunk.get("doc_id")))) continue;
                String id = chunkID(chunk);
                if (id.isEmpty()) continue;
                ids.add(id);
                fields.putIfAbsent(id, chunk);
            }
            rankings.add(ids);
            trace.add(Map.of("stage", branch + "_independent", "returned", ids.size(),
                    "durationMs", (System.nanoTime() - started) / 1e6));
        }
        List<String> order = ReciprocalRankFusion.rank(rankings, 60);
        Map<String, Double> fused = ReciprocalRankFusion.scores(rankings, 60);
        List<Map<String, Object>> chunks = new ArrayList<>();
        int begin = (request.page - 1) * request.pageSize;
        for (int i = begin; i < Math.min(begin + request.pageSize, order.size()); i++) {
            String id = order.get(i);
            var result = new LinkedHashMap<>(fields.get(id));
            result.remove("_id");
            result.put("chunk_id", id);
            result.put("similarity", fused.get(id));
            result.put("score_type", "reciprocal_rank");
            chunks.add(result);
        }
        trace.add(Map.of("stage", "independent_rank", "strategy", request.strategy, "rrfK", 60,
                "candidates", order.size(), "thresholdApplied", false));
        // RRF 不是 cosine；不能套用原版 0.2 阈值，否则几乎所有结果都会被过滤。
        return new Result(chunks, List.of(), order.size(), trace);
    }

    static void defaultsAndValidate(Request request) {
        if (request.knnTopK == null) request.knnTopK = 1024;
        if (request.knnNumCandidates == null) request.knnNumCandidates = 2048;
        if (request.similarityThreshold == null) request.similarityThreshold = 0.2;
        if (request.vectorSimilarityWeight == null)
            request.vectorSimilarityWeight = request.vectorOnly ? 1.0 : 0.3;
        if (request.rankFeature == null) request.rankFeature = Map.of("pagerank_fea", 10.0);
        if (request.aggs == null) request.aggs = true;
        if (request.highlight == null) request.highlight = false;
        if (request.allowDenseFallback == null) request.allowDenseFallback = true;
        if (request.page <= 0) request.page = 1;
        if (request.pageSize <= 0) request.pageSize = 1;
        if (request.rerankCandidatesCount == null) request.rerankCandidatesCount = 64;
        if (request.rerankCandidatesCount <= 0
                || request.page > request.rerankCandidatesCount / request.pageSize) {
            throw new IllegalArgumentException(
                    "rerank_candidates_count("
                            + request.rerankCandidatesCount
                            + ") must be greater than or equal to page("
                            + request.page
                            + ") * page_size("
                            + request.pageSize
                            + ")");
        }
        if (request.rerankCandidatesCount > 1024)
            throw new IllegalArgumentException("rerank_candidates_count must not exceed 1024");
        if (request.knnTopK > 2048)
            throw new IllegalArgumentException("knn_top_k must not exceed 2048");
        if (request.knnNumCandidates > 10000)
            throw new IllegalArgumentException("knn_num_candidates must not exceed 10000");
        if (request.rerankModel != null && request.page != 1)
            throw new IllegalArgumentException(
                    "Pagination is not supported when rerank_mdl is specified. Please set page=1 to retrieve"
                            + " the top "
                            + request.pageSize
                            + " results.");
    }

    /**
     * 将问题转换为词法表达式和向量，执行首次召回及空结果时的回退，保留每次查询 trace。
     */
    private SearchResult search(Request request, List<Map<String, Object>> trace) throws Exception {
        QueryBuilder.QuestionResult question =
                queryBuilder.question(request.question, "", minMatch(request, 0.3), request.language);
        QueryBuilder.MatchTextExpr text = question.expression();
        double[] vector = new double[0];
        ElasticsearchStore.Search result;
        if (embedding == null) {
            result = searchEngine(request, text, vector, request.similarityThreshold, trace, "fulltext");
        } else {
            vector = embedding.query(request.question);
            double similarity = request.similarityThreshold <= 0 ? 0.1 : request.similarityThreshold;
            QueryBuilder.MatchTextExpr initialText = request.vectorOnly ? null : text;
            result = searchEngine(request, initialText, vector, similarity, trace, "initial");
            // 只有零命中才进入回退；放宽词法条件后仍无结果时，可尝试纯向量召回。
            // 这里的阈值与执行顺序来自原版，不应将每个回退理解为数值上都更宽松。
            if (result.total() == 0) {
                if (request.vectorOnly || text == null) {
                    if (request.allowDenseFallback)
                        result = searchEngine(request, null, vector, 0.17, trace, "dense_fallback");
                } else {
                    QueryBuilder.MatchTextExpr relaxed =
                            queryBuilder
                                    .question(request.question, "qa", minMatch(request, 0.1), request.language)
                                    .expression();
                    result = searchEngine(request, relaxed, vector, 0.17, trace, "relaxed");
                    if (result.total() == 0 && relaxed != null && request.allowDenseFallback) {
                        result = searchEngine(request, null, vector, 0.17, trace, "dense_fallback");
                    }
                }
            }
        }
        List<Map<String, Object>> chunks = result.chunks();
        List<String> ids = new ArrayList<>();
        for (Map<String, Object> chunk : chunks) {
            String id = chunkID(chunk);
            if (!id.isEmpty()) ids.add(id);
        }
        Map<String, Map<String, Object>> fields = store.getFields(chunks, SOURCE_FIELDS);
        Set<String> keywords = new LinkedHashSet<>(question.keywords());
        for (String keyword : question.keywords()) {
            for (String fine : queryBuilder.fineGrainedKeywords(keyword, request.language)) {
                if (fine.getBytes(StandardCharsets.UTF_8).length >= 2) keywords.add(fine);
            }
        }
        Map<String, String> highlight =
                request.highlight
                        ? store.getHighlight(chunks, new ArrayList<>(keywords), "content_with_weight")
                        : Map.of();
        return new SearchResult(chunks, ids, fields, vector, highlight, result.total());
    }

    private ElasticsearchStore.Search searchEngine(
            Request request,
            QueryBuilder.MatchTextExpr text,
            double[] vector,
            double similarity,
            List<Map<String, Object>> trace,
            String stage)
            throws Exception {
        int topK = request.knnTopK <= 0 ? 1024 : request.knnTopK;
        int candidates = request.knnNumCandidates <= 0 ? 2048 : request.knnNumCandidates;
        List<String> source = new ArrayList<>(SOURCE_FIELDS);
        if (vector.length > 0) source.add("q_" + vector.length + "_vec");
        // 原 buildRetrievalFusionExpr 为 "0.001,1"；ES 实际只读第二个权重并设置 bool.boost=1-weight。
        ElasticsearchStore.Search result =
                store.search(
                        request.datasetId,
                        request.docIds,
                        text,
                        vector,
                        topK,
                        candidates,
                        request.rerankCandidatesCount,
                        similarity,
                        1,
                        request.filter,
                        request.rankFeature,
                        source);
        trace.add(
                Map.of(
                        "stage",
                        stage,
                        "request",
                        result.request(),
                        "total",
                        result.total(),
                        "returned",
                        result.chunks().size()));
        return result;
    }

    private static double minMatch(Request request, double withMatch) {
        return request.vectorSimilarityWeight >= 0.8 ? 0 : withMatch;
    }

    private Reranker.Result scoreSearchResult(
            Request request, SearchResult search, List<Map<String, Object>> trace) throws Exception {
        // 二阶段权重：词法占 1 - vectorWeight，向量（或 rerank 模型分数）占 vectorWeight。
        double vectorWeight = request.vectorSimilarityWeight;
        double tokenWeight = 1 - vectorWeight;
        // 显式配置外部 rerank 模型时优先使用它；默认路径不会自动调用模型重排。
        if (request.rerankModel != null) {
            trace.add(Map.of("stage", "rerank_model"));
            return Reranker.rerankByModel(
                    request.rerankModel,
                    search.chunks(),
                    search.ids(),
                    search.fields(),
                    request.question,
                    tokenWeight,
                    vectorWeight,
                    "content_ltks",
                    queryBuilder,
                    request.rankFeature,
                    request.language);
        }
        if (request.vectorOnly) {
            Map<String, Double> rawScores = new HashMap<>();
            for (Map<String, Object> chunk : search.chunks()) {
                for (String key : List.of("_score", "SCORE", "SIMILARITY")) {
                    if (chunk.get(key) instanceof Double score) {
                        rawScores.put(chunkID(chunk), score);
                        break;
                    }
                }
            }
            double[] vectorScores = new double[search.ids().size()];
            for (int i = 0; i < vectorScores.length; i++)
                vectorScores[i] = rawScores.getOrDefault(search.ids().get(i), 0.0);
            double[] scores = vectorScores.clone();
            Reranker.applyRankFeatureScoresForIDs(
                    search.ids(), search.fields(), scores, request.rankFeature);
            trace.add(Map.of("stage", "vector_only_scores"));
            return new Reranker.Result(scores, new double[scores.length], vectorScores);
        }
        // 默认路径只对已召回的候选二次取分，不再扩大候选池。
        // ES kNN 返回引擎 _score；异常时才改用本地 cosine，两种分数的数值含义不同。
        Map<String, Double> knn;
        try {
            knn = store.knnScores(search.chunks(), search.vector());
        } catch (Exception error) {
            trace.add(
                    Map.of("stage", "local_rerank_fallback", "reason", String.valueOf(error.getMessage())));
            return Reranker.rerankStandard(
                    search.chunks(),
                    null,
                    search.vector(),
                    request.question,
                    tokenWeight,
                    vectorWeight,
                    "content_ltks",
                    queryBuilder,
                    request.rankFeature,
                    request.language);
        }
        trace.add(Map.of("stage", "knn_scores", "scores", knn));
        return Reranker.rerankWithKNN(
                search.chunks(),
                search.ids(),
                search.fields(),
                knn,
                request.question,
                tokenWeight,
                vectorWeight,
                "content_ltks",
                queryBuilder,
                request.rankFeature,
                request.language);
    }

    /**
     * ES 与目录独立持久化，此处用目录清除过期候选，并保留原版编译文档的例外规则。
     */
    private static SearchResult prune(SearchResult result, Set<String> existingDocumentIds) {
        if (existingDocumentIds == null) return result;
        List<String> kept = new ArrayList<>();
        Map<String, Map<String, Object>> fields = new LinkedHashMap<>();
        Map<String, String> highlights = new LinkedHashMap<>();
        for (String id : result.ids()) {
            Map<String, Object> chunk = result.fields().get(id);
            if (chunk == null || chunk.isEmpty()) continue;
            String document = firstString(chunk.get("doc_id")).strip();
            String compilationDocument = string(chunk.get("doc_id"));
            boolean compiled =
                    "dataset".equals(chunk.get("scope_kwd"))
                            && !string(chunk.get("compile_kwd")).isEmpty()
                            && !compilationDocument.isEmpty()
                            && compilationDocument.equals(string(chunk.get("kb_id")));
            if (!compiled && (document.isEmpty() || !existingDocumentIds.contains(document))) continue;
            kept.add(id);
            fields.put(id, chunk);
            if (result.highlights().containsKey(id)) highlights.put(id, result.highlights().get(id));
        }
        if (kept.size() == result.ids().size()) return result;
        Map<String, Map<String, Object>> raw = new HashMap<>();
        for (Map<String, Object> chunk : result.chunks()) raw.put(chunkID(chunk), chunk);
        List<Map<String, Object>> chunks = new ArrayList<>();
        for (String id : kept) if (raw.containsKey(id)) chunks.add(raw.get(id));
        return new SearchResult(chunks, kept, fields, result.vector(), highlights, kept.size());
    }

    private static Map<String, Object> responseChunk(
            String id,
            Map<String, Object> fields,
            int position,
            Reranker.Result scores,
            SearchResult search) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("chunk_id", id);
        for (String field :
                List.of("content_ltks", "content_with_weight", "doc_id", "docnm_kwd", "kb_id")) {
            if (fields.containsKey(field)) result.put(field, fields.get(field));
        }
        String compile = firstString(fields.get("compile_kwd"));
        if (!compile.isEmpty()) result.put("compile_kwd", compile);
        result.put("image_id", fields.getOrDefault("img_id", ""));
        result.put(
                "positions", fields.get("position_int") == null ? List.of() : fields.get("position_int"));
        Object type = fields.get("doc_type_kwd");
        if (type instanceof String text && text.isEmpty()
                || type instanceof List<?> list && list.isEmpty()) type = null;
        if (!(type instanceof String) && !(type instanceof List<?>)) type = null;
        result.put("doc_type_kwd", type);
        // 三个分数用于解释排序：最终分、词法覆盖分、向量/模型分；最终分还可能含排名特征。
        result.put("similarity", scores.sim()[position]);
        result.put("term_similarity", scores.tsim()[position]);
        result.put("vector_similarity", scores.vsim()[position]);
        result.put("important_kwd", fields.getOrDefault("important_kwd", new String[0]));
        result.put("mom_id", fields.getOrDefault("mom_id", ""));
        result.put("row_id", fields.get("row_id()"));
        result.put("tag_kwd", new String[0]);
        result.put(
                "vector",
                fields.getOrDefault(
                        "q_" + search.vector().length + "_vec", new double[search.vector().length]));
        if (search.highlights().containsKey(id)) result.put("highlight", search.highlights().get(id));
        return result;
    }

    private static List<Map<String, Object>> aggregate(List<Integer> valid, SearchResult search) {
        Map<String, Map<String, Object>> groups = new LinkedHashMap<>();
        for (int index : valid) {
            if (index >= search.ids().size()) continue;
            Map<String, Object> chunk = search.fields().get(search.ids().get(index));
            if (chunk == null) continue;
            String name = string(chunk.get("docnm_kwd")), document = string(chunk.get("doc_id"));
            if (!document.isEmpty() && document.equals(chunk.get("kb_id"))) continue;
            Map<String, Object> group = groups.get(name);
            if (group == null) {
                group = new LinkedHashMap<>(Map.of("doc_name", name, "doc_id", document, "count", 0));
                groups.put(name, group);
            }
            group.put("count", ((Integer) group.get("count")) + 1);
        }
        List<Map<String, Object>> result = new ArrayList<>(groups.values());
        result.sort(Comparator.comparingInt(group -> -(Integer) group.get("count")));
        return result;
    }

    /**
     * 原 RetrievalByChildren，按父ID+文档+知识库分组，父缺失或读取失败时保留子块。
     */
    public List<Map<String, Object>> expandParents(List<Map<String, Object>> chunks) {
        if (chunks.isEmpty()) return chunks;
        Map<ParentKey, List<Map<String, Object>>> groups = new LinkedHashMap<>();
        List<Map<String, Object>> remaining = new ArrayList<>();
        for (Map<String, Object> chunk : chunks) {
            String parent = string(chunk.get("mom_id")), document = string(chunk.get("doc_id"));
            if (parent.isEmpty() || document.isEmpty()) {
                remaining.add(chunk);
                continue;
            }
            ParentKey key = new ParentKey(parent, document, string(chunk.get("kb_id")));
            groups.computeIfAbsent(key, ignored -> new ArrayList<>()).add(chunk);
        }
        if (groups.isEmpty()) return chunks;
        int vectorSize = 1024;
        for (Map.Entry<ParentKey, List<Map<String, Object>>> entry : groups.entrySet()) {
            ParentKey key = entry.getKey();
            List<Map<String, Object>> children = entry.getValue();
            Map<String, Object> parent;
            try {
                parent = store.getParent(key.parentId(), key.documentId(), key.datasetId());
            } catch (Exception error) {
                remaining.addAll(children);
                continue;
            }
            if (parent == null
                    || !key.documentId().equals(parent.get("doc_id"))
                    || !key.datasetId().equals(parent.get("kb_id"))) {
                remaining.addAll(children);
                continue;
            }
            double[] similarities =
                    children.stream()
                            .map(chunk -> chunk.get("similarity"))
                            .filter(Double.class::isInstance)
                            .mapToDouble(value -> (Double) value)
                            .toArray();
            double average = Reranker.pairwiseSum(similarities) / children.size();
            List<String> content = new ArrayList<>(), important = new ArrayList<>();
            for (Map<String, Object> child : children) {
                if (child.get("content_ltks") instanceof String text) content.add(text);
                if (child.get("important_kwd") instanceof List<?> values) {
                    for (Object value : values) if (value instanceof String text) important.add(text);
                }
            }
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("chunk_id", key.parentId());
            result.put("content_ltks", String.join(" ", content));
            for (String field : List.of("content_with_weight", "doc_id", "docnm_kwd", "kb_id"))
                result.put(field, parent.get(field));
            result.put("important_kwd", important);
            result.put("image_id", parent.get("img_id") == null ? "" : parent.get("img_id"));
            result.put("similarity", average);
            result.put("vector_similarity", average);
            result.put("term_similarity", average);
            result.put("positions", parent.get("position_int"));
            result.put("doc_type_kwd", string(parent.get("doc_type_kwd")));
            result.put("vector", new double[vectorSize]);
            boolean vectorFound = false;
            for (Map<String, Object> child : children) {
                for (Map.Entry<String, Object> field : child.entrySet()) {
                    if (field.getKey().endsWith("_vec") && field.getValue() instanceof double[] vector) {
                        result.put("vector", vector);
                        vectorSize = vector.length;
                        vectorFound = true;
                        break;
                    }
                }
                if (vectorFound) break;
            }
            remaining.add(result);
        }
        // 原算法是交换排序，保持其非稳定的相等分数行为。
        for (int i = 0; i < remaining.size(); i++) {
            for (int j = i + 1; j < remaining.size(); j++) {
                double left = remaining.get(i).get("similarity") instanceof Double score ? score : 0;
                double right = remaining.get(j).get("similarity") instanceof Double score ? score : 0;
                if (right > left) Collections.swap(remaining, i, j);
            }
        }
        return remaining;
    }

    private static String chunkID(Map<String, Object> chunk) {
        String id = string(chunk.get("id"));
        return id.isEmpty() ? string(chunk.get("_id")) : id;
    }

    private static String string(Object value) {
        return value instanceof String text ? text : "";
    }

    private static String firstString(Object value) {
        if (value instanceof String text) return text;
        if (value instanceof List<?> values && !values.isEmpty()) return string(values.getFirst());
        if (value instanceof String[] values && values.length > 0) return values[0];
        return "";
    }

    private static Result empty(List<Map<String, Object>> trace) {
        return new Result(List.of(), List.of(), 0, trace);
    }
}
