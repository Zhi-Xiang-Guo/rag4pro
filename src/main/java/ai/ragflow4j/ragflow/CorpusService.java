package ai.ragflow4j.ragflow;

import ai.ragflow4j.Settings;

import ai.ragflow4j.EmbeddingClient;

import java.nio.charset.StandardCharsets;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.*;

import net.jpountz.xxhash.XXHashFactory;

/**
 * 标准文本 RAG 的业务编排入口，把 HTTP 参数连接到各阶段的算法和外部服务。
 *
 * <p>三条主线：index 将文档写成索引，retrieve 返回相关块，answer 在检索后生成带引用的回答。 初次阅读建议先跟随 retrieve → retrievalRequest →
 * StandardRetrieval.retrieve。
 *
 * <p>公开业务方法使用 synchronized，让本演示服务内的目录和索引更新串行执行；跨 ES 与本地目录的持久化仍需显式补偿。
 */
public final class CorpusService {
    private final ElasticsearchStore store;
    private final DocumentCatalog catalog;
    private final DocumentSources sources;
    private final IndexingPipeline indexing;
    private final ChunkingStrategies chunker;
    private final Cl100k tokens;
    private final StandardRetrieval retrieval;
    private final ChatPipeline chat;
    private final ExtractiveChat extractiveChat;
    private final QueryRefiner refiner;
    private final EmbeddingClient embeddings;
    private final LocalChatModel chatModel;
    private final LocalChatModel extractiveModel;
    private final int contextLength;
    private final String embeddingIdentity;
    private final ContentVectorCache contentCache;
    private Reranker.Model reranker;

    public CorpusService(
            ElasticsearchStore store,
            DocumentCatalog catalog,
            NativeTokenizer lexical,
            EmbeddingClient embeddings,
            IndexingPipeline.Embedder indexEmbedder,
            LocalChatModel chatModel,
            int contextLength) {
        this(store, catalog, lexical, embeddings, indexEmbedder, chatModel, contextLength, null);
    }

    public CorpusService(
            ElasticsearchStore store,
            DocumentCatalog catalog,
            NativeTokenizer lexical,
            EmbeddingClient embeddings,
            IndexingPipeline.Embedder indexEmbedder,
            LocalChatModel chatModel,
            int contextLength,
            ContentVectorCache contentCache) {
        this(store, catalog, lexical, embeddings, indexEmbedder, chatModel, contextLength, contentCache, chatModel);
    }

    public CorpusService(
            ElasticsearchStore store,
            DocumentCatalog catalog,
            NativeTokenizer lexical,
            EmbeddingClient embeddings,
            IndexingPipeline.Embedder indexEmbedder,
            LocalChatModel chatModel,
            int contextLength,
            ContentVectorCache contentCache,
            LocalChatModel extractiveModel) {
        this.store = store;
        this.catalog = catalog;
        sources = new DocumentSources(catalog.directory().resolve("sources"));
        this.embeddings = embeddings;
        this.chatModel = chatModel;
        this.extractiveModel = extractiveModel;
        this.contextLength = contextLength;
        this.contentCache = contentCache;
        this.embeddingIdentity = embeddings.identity() + "|ragflow-title-blend-v1";
        Cl100k counter = new Cl100k();
        tokens = counter;
        indexing =
                new IndexingPipeline(lexical, counter, indexEmbedder, contentCache, embeddings.identity());
        chunker = new ChunkingStrategies(counter);
        retrieval =
                new StandardRetrieval(
                        store, new QueryBuilder(lexical), new QueryEmbeddingCache(
                                q -> embeddings.embed(List.of(q)).getFirst(),
                                Integer.parseInt(Settings.get("QUERY_CACHE_ENTRIES", "0")),
                                Duration.ofMinutes(30)));
        chat =
                new ChatPipeline(
                        chatModel,
                        counter,
                        embeddings::embed,
                        (chunks, dimension) -> {
                            for (var chunk : chunks) {
                                String id = Objects.toString(chunk.get("chunk_id"), "");
                                Map<String, Object> stored = store.get(id);
                                if (stored != null && stored.containsKey("q_" + dimension + "_vec"))
                                    chunk.put("vector", vector(stored.get("q_" + dimension + "_vec")));
                            }
                        });
        refiner = new QueryRefiner(chatModel);
        extractiveChat = new ExtractiveChat(extractiveModel::selectEvidence, counter, extractiveModel.contextLength());
    }

    public synchronized void configureReranker(Reranker.Model model) {
        reranker = model;
    }

    /**
     * 文档入库主线：解析 → 分块 → 分词与向量化 → 字段整理 → ES 和文档目录持久化。
     */
    public synchronized Map<String, Object> index(Map<String, Object> request) throws Exception {
        String dataset = required(request, "datasetId"),
                externalId = required(request, "docId"),
                name = required(request, "title"),
                text = required(request, "text");
        // 用户传的 docId 只在数据集内唯一；内部 ID 同时包含数据集，避免不同数据集互相覆盖。
        String docId = DocumentCatalog.identity(dataset, externalId),
                language = string(request, "language", importLanguage(dataset));
        var plan = chunkPlan(request, dataset, name, text);
        var config = plan.config();
        List<String> warnings = new ArrayList<>(plan.warnings());
        var lexicalConfig =
                new IndexingPipeline.Config(
                        language,
                        true,
                        true,
                        number(request, "filenameEmbeddingWeight", 0.1),
                        strings(request, "embeddingFields", List.of("text")));
        // 原 chunker/register.go 在 Tokenizer 之前盖章，缓存依赖同一个确定性 chunk ID。
        assignChunkIds(plan, docId);
        String taskId = UUID.randomUUID().toString();
        List<Map<String, Object>> chunks =
                new ArrayList<>(indexing.process(name, plan.pieces(), lexicalConfig, taskId));
        if (chunks.isEmpty()) throw new IllegalArgumentException("没有可索引的文本块");
        for (var entry : catalog.list(dataset))
            if (!embeddingIdentity.equals(entry.get("embeddingIdentity")))
                throw new IllegalArgumentException("该数据集使用不同的 embedding 配置，请创建新的数据集");
        materialize(dataset, docId, name, chunks, integer(request, "pagerank", 0));
        int sourceCount =
                (int)
                        chunks.stream()
                                .filter(c -> ((Number) c.get("available_int")).intValue() == 1)
                                .map(c -> c.get("id"))
                                .distinct()
                                .count();
        int parentCount =
                (int)
                        chunks.stream()
                                .filter(c -> ((Number) c.get("available_int")).intValue() == 0)
                                .map(c -> c.get("id"))
                                .distinct()
                                .count();
        var old = catalog.get(dataset, externalId);
        Map<String, Object> details =
                new LinkedHashMap<>(
                        Map.of(
                                "title",
                                name,
                                "language",
                                language,
                                "chunkCount",
                                sourceCount,
                                "sourceSha256",
                                DocumentCatalog.sourceDigest(text.getBytes(StandardCharsets.UTF_8)),
                                "embeddingIdentity",
                                embeddingIdentity,
                                "chunkStrategy", plan.strategy(),
                                "chunkTokenSize", config.chunkTokenSize(),
                                "overlapPercent", config.overlapPercent()));
        if (request.containsKey("metadata")) {
            if (!(request.get("metadata") instanceof Map<?, ?> metadata))
                throw new IllegalArgumentException("metadata 必须为对象");
            details.put("metadata", objectMap(metadata));
        } else if (old != null && old.containsKey("metadata"))
            details.put("metadata", old.get("metadata"));
        List<Map<String, Object>> oldChunks = store.allDocumentChunks(dataset, docId);
        // ES 和本地文档目录没有共享事务：先保留旧块，目录写失败时尝试恢复 ES。
        store.replace(dataset, docId, chunks);
        try {
            catalog.put(dataset, externalId, details);
        } catch (Exception failure) {
            try {
                store.replace(dataset, docId, oldChunks);
            } catch (Exception rollback) {
                failure.addSuppressed(rollback);
            }
            throw failure;
        }
        if (contentCache != null) {
            try {
                contentCache.purgeTask(taskId);
            } catch (java.io.IOException error) {
                warnings.add("文档已持久化，任务缓存清理失败，将按 TTL 过期");
            }
        }
        return Map.of(
                "datasetId",
                dataset,
                "docId",
                externalId,
                "internalDocId",
                docId,
                "chunkCount",
                sourceCount,
                "parentCount",
                parentCount,
                "chunks",
                summaries(chunks),
                "warnings",
                warnings,
                "trace",
                Map.of(
                        "chunkPolicy",
                        plan.strategy().equals("general") ? "GeneralChunker" : plan.strategy(),
                        "chunkTokenSize",
                        config.chunkTokenSize(),
                        "overlapPercent",
                        config.overlapPercent(),
                        "filenameEmbeddingWeight",
                        lexicalConfig.filenameEmbeddingWeight(),
                        "tokenizer",
                        "RAGAnalyzer",
                        "source",
                        "e4bd0801d5169c3dc0847ce0ea06fc6c3e516105"));
    }

    /** 有界新增批次：相同 chunker/tokenizer/embedding/字段；一次 bulk 和目录原子落盘。 */
    public synchronized Map<String, Object> indexBatch(Map<String, Object> input) throws Exception {
        String dataset = required(input, "datasetId");
        if (!(input.get("documents") instanceof List<?> raw) || raw.isEmpty() || raw.size() > 64)
            throw new IllegalArgumentException("documents requires 1..64 new documents");
        List<Map<String, Object>> requests = new ArrayList<>();
        List<IndexingPipeline.Document> prepared = new ArrayList<>();
        List<ChunkingStrategies.Plan> plans = new ArrayList<>();
        Set<String> unique = new HashSet<>();
        for (Object value : raw) {
            if (!(value instanceof Map<?, ?> map)) throw new IllegalArgumentException("document must be an object");
            Map<String, Object> request = objectMap(map);
            String id = required(request, "docId"), name = required(request, "title"), text = required(request, "text");
            if (!unique.add(id) || catalog.get(dataset, id) != null)
                throw new IllegalArgumentException("batch accepts unique new docIds only: " + id);
            var plan = chunkPlan(request, dataset, name, text);
            if (plan.pieces().isEmpty()) throw new IllegalArgumentException("empty document");
            String internal = DocumentCatalog.identity(dataset, id);
            assignChunkIds(plan, internal);
            plans.add(plan);
            prepared.add(new IndexingPipeline.Document(name, plan.pieces(),
                    new IndexingPipeline.Config(string(request, "language", importLanguage(dataset)), true, true,
                            number(request, "filenameEmbeddingWeight", 0.1), strings(request, "embeddingFields", List.of("text")))));
            if (request.containsKey("metadata") && !(request.get("metadata") instanceof Map<?, ?>))
                throw new IllegalArgumentException("metadata must be an object");
            requests.add(request);
        }
        for (var entry : catalog.list(dataset))
            if (!embeddingIdentity.equals(entry.get("embeddingIdentity")))
                throw new IllegalArgumentException("embedding configuration mismatch");
        var processed = indexing.processBatch(prepared);
        List<Map<String, Object>> all = new ArrayList<>();
        Map<String, Map<String, Object>> entries = new LinkedHashMap<>();
        for (int i = 0; i < processed.size(); i++) {
            var request = requests.get(i);
            String id = required(request, "docId");
            var chunks = new ArrayList<>(processed.get(i));
            materialize(dataset, DocumentCatalog.identity(dataset, id), required(request, "title"), chunks,
                    integer(request, "pagerank", 0));
            int count = (int) chunks.stream().filter(c -> ((Number)c.get("available_int")).intValue() == 1)
                    .map(c -> c.get("id")).distinct().count();
            String body = required(request, "text");
            Map<String, Object> details = new LinkedHashMap<>(Map.of("title", required(request, "title"),
                    "language", prepared.get(i).config().language(), "chunkCount", count, "embeddingIdentity", embeddingIdentity,
                    "sourceSha256", DocumentCatalog.sourceDigest(body.getBytes(StandardCharsets.UTF_8)),
                    "chunkStrategy", plans.get(i).strategy(), "chunkTokenSize", plans.get(i).config().chunkTokenSize(),
                    "overlapPercent", plans.get(i).config().overlapPercent()));
            if (request.containsKey("metadata")) details.put("metadata", request.get("metadata"));
            entries.put(id, details);
            all.addAll(chunks);
        }
        store.createBatch(dataset, all);
        try { catalog.putNewBatch(dataset, entries); }
        catch (Exception failure) {
            try { store.removeBatch(all); } catch (Exception rollback) { failure.addSuppressed(rollback); }
            throw failure;
        }
        return Map.of("datasetId", dataset, "documentCount", entries.size(), "chunkCount", all.size(),
                "docIds", List.copyOf(entries.keySet()));
    }

    private ChunkingStrategies.Plan chunkPlan(Map<String, Object> request, String dataset, String title, String text) {
        var options = catalog.dataset(dataset);
        String savedStrategy = string(options, "chunkStrategy", "general");
        String strategy = string(request, "chunkStrategy", savedStrategy);
        double overlap = strategy.equals(savedStrategy) ? number(options, "overlapPercent", 0) : 0;
        var defaults = GeneralChunker.Config.defaults();
        var config = new GeneralChunker.Config(
                integer(request, "chunkTokenSize", integer(options, "chunkTokenSize", defaults.chunkTokenSize())),
                strings(request, "delimiters", defaults.delimiters()),
                strings(request, "childrenDelimiters", List.of()),
                number(request, "overlapPercent", overlap));
        String format = string(request, "fileType", title.toLowerCase(Locale.ROOT).endsWith(".md") ? "md" : "txt");
        return chunker.plan(text, format, strategy, config);
    }

    private void assignChunkIds(ChunkingStrategies.Plan plan, String documentId) {
        for (int i = 0; i < plan.pieces().size(); i++) {
            var piece = plan.pieces().get(i);
            String text = Objects.toString(piece.get("text"), "");
            String key = plan.strategy().equals("general") ? text + documentId
                    : documentId + "\0" + plan.strategy() + "\0" + i + "\0" + text;
            piece.putIfAbsent("id", hash(key));
        }
    }

    public synchronized Map<String, Object> previewChunks(Map<String, Object> request) {
        var plan = chunkPlan(request, string(request, "datasetId", ""), required(request, "title"), required(request, "text"));
        List<Map<String, Object>> preview = plan.pieces().stream().limit(30).map(piece -> Map.of(
                "text", piece.get("text"), "tokens", piece.getOrDefault("tk_nums", 0),
                "section", piece.getOrDefault("section_kwd", ""), "hasParent", piece.containsKey("mom"))).toList();
        return Map.of("chunkStrategy", plan.strategy(), "total", plan.pieces().size(),
                "parentCount", plan.pieces().stream().filter(c -> c.containsKey("mom"))
                        .map(c -> c.getOrDefault("parent_ref", c.get("mom"))).distinct().count(),
                "sectionCount", plan.pieces().stream().map(c -> c.get("section_ref")).filter(Objects::nonNull).distinct().count(),
                "pieces", preview, "warnings", plan.warnings());
    }

    /**
     * 将解析阶段的临时字段整理为 ES 字段，并为子块创建不参与常规召回的父块实体。
     */
    private void materialize(
            String dataset, String docId, String title, List<Map<String, Object>> chunks, int pagerank) {
        String time =
                ZonedDateTime.now(ZoneId.of("Asia/Shanghai"))
                        .format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        double timestamp = Instant.now().toEpochMilli() / 1000.0;
        List<Map<String, Object>> parents = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        int order = 1 << 30;
        for (var c : chunks) {
            String text = Objects.toString(c.remove("text"), "");
            c.putIfAbsent("id", hash(text + docId));
            c.put("doc_id", docId);
            c.put("kb_id", dataset);
            c.put("docnm_kwd", title);
            c.put("create_time", time);
            c.put("create_timestamp_flt", timestamp);
            c.put("content_with_weight", text);
            c.put("available_int", 1);
            if (pagerank != 0) c.put("pagerank_fea", pagerank);
            for (String consumed :
                    List.of(
                            "questions",
                            "keywords",
                            "summary",
                            "tk_nums",
                            "file_type",
                            "output_format",
                            "layout",
                            "layout_type",
                            "layoutno",
                            "image",
                            "media_ref",
                            "context_above",
                            "context_below",
                            "page_number",
                            "_preserve_order",
                            "table_id",
                            "sheet",
                            "sheet_index",
                            "headers",
                            "cells",
                            "row_start",
                            "row_end",
                            "col_start",
                            "col_end",
                            "metadata"))
                c.remove(consumed);
            Object docType = c.remove("doc_type");
            if (docType != null) c.put("doc_type_kwd", docType);
            String mom = Objects.toString(c.remove("mom"), "");
            Object fallbackMom = c.remove("mom_with_weight");
            if (mom.isBlank() && fallbackMom instanceof String textMom) mom = textMom;
            Object parentRef = c.remove("parent_ref");
            String section = Objects.toString(c.remove("section_context"), "");
            Object sectionRef = c.remove("section_ref");
            if (!section.isBlank()) {
                String id = hash(dataset + "\0" + docId + "\0section\0" + sectionRef + "\0" + section);
                c.put("section_id", id);
                if (seen.add(id)) parents.add(contextChunk(c, id, section, "section", order++));
            }
            if (!mom.isBlank()) {
                String identity = parentRef == null ? mom : "parent\0" + parentRef + "\0" + mom;
                String id = hash(dataset + "\0" + docId + "\0" + identity);
                c.put("mom_id", id);
                if (seen.add(id)) parents.add(contextChunk(c, id, mom, "parent", order++));
            }
        }
        chunks.addAll(parents);
    }

    private Map<String, Object> contextChunk(Map<String, Object> leaf, String id, String content, String level, int order) {
        Map<String, Object> context = new LinkedHashMap<>();
        for (String key : List.of("doc_id", "docnm_kwd", "kb_id", "position_int", "create_timestamp_flt",
                "page_num_int", "top_int", "section_kwd", "section_id"))
            if (leaf.containsKey(key)) context.put(key, leaf.get(key));
        context.put("id", id);
        context.put("content_with_weight", content);
        context.put("available_int", 0);
        context.put("chunk_order_int", order);
        context.put("context_level_kwd", level);
        return context;
    }

    /**
     * 直接检索只返回相关块和评分；父块回填、上下文拼接与模型回答由 answer 继续处理。
     */
    public synchronized StandardRetrieval.Result retrieve(Map<String, Object> input)
            throws Exception {
        return retrieval.retrieve(retrievalRequest(input, false));
    }

    public synchronized Map<String, Object> retrieveContext(Map<String, Object> input) throws Exception {
        var request = retrievalRequest(input, false);
        long started = System.nanoTime();
        var retrieved = retrieval.retrieve(request);
        double retrievalMs = (System.nanoTime() - started) / 1e6;
        var evidence = collectEvidence(input, request.datasetId, retrieved.chunks(), 1600);
        // Use copies because leaf mode can share the original hit objects.
        var hits = retrieved.chunks().stream().map(c -> (Map<String, Object>) new LinkedHashMap<>(c)).toList();
        enrichDocumentMetadata(request.datasetId, hits);
        return Map.of("query", request.question, "total", retrieved.total(), "retrievalMs", retrievalMs,
                "trace", retrieved.trace(), "hits", hits.stream().map(CitationService::formatChunk).toList(),
                "evidence", evidence.chunks().stream().map(CitationService::formatChunk).toList(),
                "evidenceTokens", evidence.tokens(), "contextMode", evidence.mode());
    }

    private record Evidence(List<Map<String, Object>> chunks, int tokens, String mode) {}

    private Evidence collectEvidence(Map<String, Object> input, String dataset,
                                     List<Map<String, Object>> hits, int defaultBudget) {
        String mode = string(input, "contextMode", "parent");
        int budget = integer(input, "evidenceBudget", defaultBudget);
        if (budget < 0 || budget > 8192) throw new IllegalArgumentException("证据预算须为 0–8192 tokens");
        var candidates = new ArrayDeque<>(switch (mode) {
            case "leaf" -> hits.stream().map(StandardRetrieval.ContextCandidate::leaf).toList();
            case "parent" -> retrieval.contextCandidates(hits, false);
            case "section" -> retrieval.contextCandidates(hits, true);
            default -> throw new IllegalArgumentException("未知证据扩展层级");
        });
        List<Map<String, Object>> packed = new ArrayList<>();
        int used = 0;
        while (!candidates.isEmpty()) {
            var candidate = candidates.removeFirst();
            var chunk = candidate.chunk();
            int size = tokens.count(ContextBuilder.text(chunk, "content_with_weight", "content"));
            if (budget == 0 || size <= budget - used) {
                packed.add(new LinkedHashMap<>(chunk));
                used += size;
            } else {
                // Try a smaller context at this rank before moving on to unrelated hits.
                var smaller = candidate.section()
                        ? retrieval.contextCandidates(candidate.children(), false)
                        : candidate.children().stream().map(StandardRetrieval.ContextCandidate::leaf).toList();
                smaller.reversed().forEach(candidates::addFirst);
            }
        }
        enrichDocumentMetadata(dataset, packed);
        return new Evidence(packed, used, mode);
    }

    /**
     * 将 HTTP 字段转换为内部检索请求，并确定数据集、文档范围和评分配置。
     *
     * <p>chatting 只影响这里的默认 topK 和 threshold；两种入口共用后续检索算法。
     */
    StandardRetrieval.Request retrievalRequest(Map<String, Object> input, boolean chatting) {
        var request = new StandardRetrieval.Request();
        request.datasetId = required(input, "datasetId");
        request.question = required(input, "question");
        String doc = string(input, "docId", "");
        request.docIds =
                doc.isBlank() ? List.of() : List.of(DocumentCatalog.identity(request.datasetId, doc));
        if (input.get("docIds") instanceof List<?> requestedIds) {
            Set<String> known = catalog.existingIds(request.datasetId);
            request.docIds =
                    requestedIds.stream()
                            .map(Object::toString)
                            .map(id -> known.contains(id) ? id : DocumentCatalog.identity(request.datasetId, id))
                            .toList();
        }
        // metadata 先筛选文档 ID；后面的 ES 查询再在这些文档中查 chunk。
        request.docIds =
                MetadataFilter.filterDocumentIds(
                        MetadataFilter.configFromRequest(input),
                        catalog.list(request.datasetId),
                        request.question,
                        chatModel,
                        request.docIds);
        request.language = string(input, "language", datasetLanguage(request.datasetId));
        request.page = integer(input, "page", 1);
        // topK 是最终返回数量，candidateK 是供二阶段评分的候选池，两者不要混为一谈。
        request.pageSize = integer(input, "topK", chatting ? 6 : 5);
        request.rerankCandidatesCount = integer(input, "candidateK", 64);
        request.knnTopK = integer(input, "knnTopK", 1024);
        request.knnNumCandidates = integer(input, "knnNumCandidates", 2048);
        request.similarityThreshold = number(input, "threshold", chatting ? 0.1 : 0.2);
        String mode = string(input, "mode", "hybrid");
        request.strategy = string(input, "strategy", "standard");
        if (!Set.of("standard", "lexical", "dense", "rrf", "rrf_diverse").contains(request.strategy))
            throw new IllegalArgumentException("unknown strategy");
        request.vectorOnly = mode.equals("vector");
        if (!Set.of("vector", "bm25", "hybrid").contains(mode))
            throw new IllegalArgumentException("未知检索模式");
        request.vectorSimilarityWeight =
                mode.equals("bm25") ? 0 : mode.equals("vector") ? 1 : number(input, "vectorWeight", 0.3);
        if (request.similarityThreshold < 0 || request.similarityThreshold > 1)
            throw new IllegalArgumentException("threshold 必须在 0 到 1 之间");
        if (request.vectorSimilarityWeight < 0 || request.vectorSimilarityWeight > 1)
            throw new IllegalArgumentException("vectorWeight 必须在 0 到 1 之间");
        if (request.knnTopK < 1 || request.knnTopK > 2048)
            throw new IllegalArgumentException("knnTopK 必须在 1 到 2048 之间");
        if (request.knnNumCandidates < request.knnTopK || request.knnNumCandidates > 10000)
            throw new IllegalArgumentException("knnNumCandidates 必须不小于 knnTopK 且不超过 10000");
        // 目录提供源文档的存活集合，检索器据此排除 ES 中可能残留的已删除文档块。
        request.existingDocumentIds = catalog.existingIds(request.datasetId);
        request.filter = input.get("filter") instanceof Map<?, ?> map ? objectMap(map) : Map.of();
        request.allowDenseFallback = bool(input, "allowDenseFallback", true);
        request.highlight = bool(input, "highlight", false);
        request.aggs = bool(input, "aggs", true);
        if (input.containsKey("rankFeature")) {
            if (!(input.get("rankFeature") instanceof Map<?, ?> features))
                throw new IllegalArgumentException("rankFeature 必须为对象");
            Map<String, Double> rank = new LinkedHashMap<>();
            for (var feature : features.entrySet()) {
                if (!(feature.getValue() instanceof Number weight)
                        || !Double.isFinite(weight.doubleValue()))
                    throw new IllegalArgumentException("rankFeature 权重必须为有限数值");
                rank.put(feature.getKey().toString(), weight.doubleValue());
            }
            request.rankFeature = rank;
        }
        if (bool(input, "useReranker", false)) {
            if (reranker == null) throw new IllegalArgumentException("尚未配置 rerank 模型");
            request.rerankModel = reranker;
        }
        return request;
    }

    // 对应 entity.KnowledgebasesLanguage，相同语言按首个原值返回，混合语言回退 English。
    String datasetLanguage(String dataset) {
        List<Map<String, Object>> documents = catalog.list(dataset);
        if (documents.isEmpty()) return "";
        String language = Objects.toString(documents.getFirst().get("language"), "");
        for (var document : documents) {
            String current = Objects.toString(document.get("language"), "");
            if (!current.strip().equalsIgnoreCase(language.strip())) return "";
        }
        return language;
    }

    private String importLanguage(String dataset) {
        return string(catalog.dataset(dataset), "language", datasetLanguage(dataset));
    }

    void enrichDocumentMetadata(String dataset, List<Map<String, Object>> chunks) {
        Map<String, Map<String, Object>> documents = new HashMap<>();
        for (var document : catalog.list(dataset))
            documents.put(document.get("internalDocId").toString(), document);
        for (var chunk : chunks) {
            if (!dataset.equals(chunk.get("kb_id"))) continue;
            Map<String, Object> document = documents.get(Objects.toString(chunk.get("doc_id"), ""));
            if (document == null) continue;
            chunk.put("document_external_id", document.get("docId"));
            if (document.get("metadata") instanceof Map<?, ?> metadata && !metadata.isEmpty()) {
                chunk.put("document_metadata", objectMap(metadata));
            }
        }
    }

    /**
     * 问答主线：整理问题 → 检索 → 父块回填 → metadata 补充 → 上下文预算 → 模型与引用。
     */
    public synchronized Map<String, Object> answer(Map<String, Object> input) throws Exception {
        String answerMode = string(input, "answerMode", "standard");
        if (!Set.of("standard", "extractive").contains(answerMode))
            throw new IllegalArgumentException("answerMode 必须为 standard 或 extractive");
        boolean extractive = answerMode.equals("extractive");
        List<ContextBuilder.Message> messages = new ArrayList<>();
        if (input.get("messages") instanceof List<?> list) {
            for (Object raw : list) {
                if (!(raw instanceof Map<?, ?> m)) throw new IllegalArgumentException("消息格式无效");
                messages.add(
                        new ContextBuilder.Message(
                                Objects.toString(m.get("role"), ""), Objects.toString(m.get("content"), "")));
            }
        } else messages.add(new ContextBuilder.Message("user", required(input, "question")));
        if (messages.isEmpty() || !messages.getLast().role().equals("user"))
            throw new IllegalArgumentException("最后一条消息必须来自 user");
        List<String> recentQuestions =
                messages.stream()
                        .filter(message -> message.role().equals("user"))
                        .map(ContextBuilder.Message::content)
                        .toList();
        recentQuestions = recentQuestions.subList(Math.max(0, recentQuestions.size() - 3), recentQuestions.size());
        String question = String.join(" ", recentQuestions);
        String latestQuestion = recentQuestions.getLast();
        boolean rewrite = bool(input, "refineMultiturn", !extractive) && recentQuestions.size() > 1;
        List<String> questionsForEvidence = recentQuestions;
        List<String> warnings = new ArrayList<>();
        if (rewrite) {
            var refined = extractive ? QueryRefiner.standaloneFollowUp(recentQuestions, extractiveModel)
                    : refiner.fullQuestionOutcome(messages, "");
            if (refined.error() == null && !refined.text().isBlank()) {
                question = refined.text();
            } else if (extractive) {
                question = latestQuestion;
                warnings.add("追问改写失败，已检索当前问题");
            } else {
                warnings.add("多轮问题改写失败，已保留最近三个用户问题");
            }
            if (extractive) questionsForEvidence = List.of(question);
        }
        List<String> languages = strings(input, "crossLanguages", List.of());
        if (!languages.isEmpty()) question = refiner.crossLanguages(question, languages);
        Map<String, Object> retrievalInput = new LinkedHashMap<>(input);
        retrievalInput.put("question", question);
        if (input.get("messages") instanceof List<?> rawMessages
                && !rawMessages.isEmpty()
                && rawMessages.getLast() instanceof Map<?, ?> lastMessage
                && lastMessage.get("doc_ids") instanceof List<?> messageDocIds) {
            retrievalInput.put("docIds", messageDocIds);
        }
        var searchRequest = retrievalRequest(retrievalInput, true);
        // 原 Phase 8 顺序是跨语言 → metadata 过滤 → keyword 增强。
        if (bool(input, "keyword", false)) question = refiner.appendKeywords(question);
        searchRequest.question = question;
        long retrievalStarted = System.nanoTime();
        var retrieved = retrieval.retrieve(searchRequest);
        double retrievalMs = (System.nanoTime() - retrievalStarted) / 1e6;
        var evidence = collectEvidence(input, searchRequest.datasetId, retrieved.chunks(), 0);
        var expanded = evidence.chunks();
        var sources = new ChatPipeline.Retrieval(expanded, retrieved.docAggs());
        boolean quote = bool(input, "quote", true);
        ChatPipeline.Result result;
        if (extractive) {
            String refusal = latestQuestion.codePoints().anyMatch(cp -> Character.UnicodeScript.of(cp) == Character.UnicodeScript.HAN)
                    ? "资料不足，无法回答。" : "No supporting evidence was found in the knowledge base.";
            result = extractiveChat.complete(questionsForEvidence, sources, refusal, quote);
        } else {
            result = generateAnswer(input, messages, sources, quote);
        }
        Map<String, Object> output = new LinkedHashMap<>();
        output.put("answer", result.answer());
        output.put("reasoning", result.reasoning());
        output.put("reference", result.reference());
        output.put("usage", result.usage());
        output.put("query", question);
        output.put("model", extractive ? extractiveModel.model() : chatModel.model());
        output.put("prompt", result.prompt());
        output.put("retrievalTrace", retrieved.trace());
        output.put("warnings", warnings);
        if (input.containsKey("contextMode") || input.containsKey("evidenceBudget")) {
            output.put("contextMode", evidence.mode());
            output.put("evidenceTokens", evidence.tokens());
        }
        if (extractive) {
            output.put("answerMode", answerMode);
            output.put("answerStatus", result.emptyResponse() ? "insufficient_evidence" : "answered");
        }
        if (bool(input, "evaluationTrace", false)) {
            output.put("evaluation", Map.of("retrievalMs", retrievalMs, "modelBoundary", result.audit(),
                    "citedIndices", result.citedIndices(), "emptyResponse", result.emptyResponse(),
                    "retrievedChunks", expanded.stream().map(CitationService::formatChunk).toList()));
        }
        return output;
    }

    private ChatPipeline.Result generateAnswer(Map<String, Object> input, List<ContextBuilder.Message> messages,
                                               ChatPipeline.Retrieval sources, boolean quote) throws Exception {
        String system =
                string(
                        input,
                        "systemPrompt",
                        "You are an intelligent assistant. Please summarize the content of the dataset to"
                                + " answer the question. Please list the data in the dataset and answer in detail."
                                + " When all dataset content is irrelevant to the question, your answer must"
                                + " include the sentence \"The answer you are looking for is not found in the"
                                + " dataset!\" Answers need to consider chat history.\n"
                                + "      Here is the knowledge base:\n"
                                + "      {knowledge}\n"
                                + "      The above is the knowledge base.");
        Map<String, Object> generation =
                input.get("generation") instanceof Map<?, ?> map
                        ? objectMap(map)
                        : Map.of("max_tokens", 1024);
        var config =
                new ChatPipeline.Request(
                        messages,
                        system,
                        null,
                        Map.of(),
                        generation,
                        contextLength,
                        string(
                                input,
                                "emptyResponse",
                                "Sorry! No relevant content was found in the knowledge base!"),
                        quote,
                        true,
                        "",
                        false);
        return chat.complete(config, sources);
    }

    public DocumentSources sources() {
        return sources;
    }

    public Map<String, Object> document(String dataset, String docId) {
        var document = catalog.get(dataset, docId);
        return document == null ? null : new LinkedHashMap<>(document);
    }

    public synchronized List<Map<String, Object>> documents(String dataset) {
        return catalog.list(dataset);
    }

    public synchronized List<Map<String, Object>> datasets() {
        return catalog.datasets();
    }

    public synchronized Map<String, Object> saveDataset(Map<String, Object> input) throws Exception {
        String id = required(input, "datasetId"), name = required(input, "name").strip();
        if (!id.matches("[a-z0-9][a-z0-9_-]{0,63}") || name.length() > 100)
            throw new IllegalArgumentException("知识库 ID 或名称超出允许范围");
        String strategy = string(input, "chunkStrategy", "parent_child");
        String language = string(input, "language", "");
        if (!Set.of("", "Chinese", "English").contains(language))
            throw new IllegalArgumentException("请选择自动检测、中文或英文");
        var config = new GeneralChunker.Config(integer(input, "chunkTokenSize", 160), List.of(),
                List.of(), number(input, "overlapPercent", 0));
        ChunkingStrategies.validate(strategy, config);
        catalog.putDataset(id, Map.of("name", name, "chunkStrategy", strategy, "language", language,
                "chunkTokenSize", config.chunkTokenSize(), "overlapPercent", config.overlapPercent()));
        return catalog.datasets().stream().filter(d -> id.equals(d.get("datasetId"))).findFirst().orElseThrow();
    }

    public synchronized void deleteEmptyDataset(String id) throws Exception {
        catalog.deleteEmptyDataset(id);
    }

    public synchronized List<Map<String, Object>> chunks(String dataset, String doc)
            throws Exception {
        return summaries(
                store.chunks(dataset, doc == null ? null : DocumentCatalog.identity(dataset, doc)));
    }

    public synchronized void delete(String dataset, String doc) throws Exception {
        String internalId = DocumentCatalog.identity(dataset, doc);
        List<Map<String, Object>> oldChunks = store.allDocumentChunks(dataset, internalId);
        try {
            store.deleteDocument(dataset, internalId);
            catalog.delete(dataset, doc);
        } catch (Exception failure) {
            try {
                store.replace(dataset, internalId, oldChunks);
            } catch (Exception rollback) {
                failure.addSuppressed(rollback);
            }
            throw failure;
        }
    }

    public Map<String, Object> health() throws Exception {
        return Map.of(
                "status",
                "ok",
                "index",
                store.health(),
                "chatModel",
                chatModel.model(),
                "pipeline",
                "ragflow-standard",
                "embeddingIdentity", embeddings.identity(),
                "queryCacheEntries", Integer.parseInt(Settings.get("QUERY_CACHE_ENTRIES", "0")),
                "chatContext", contextLength,
                "extractiveModel", extractiveModel.model(),
                "extractiveContext", extractiveModel.contextLength());
    }

    public static List<Map<String, Object>> summaries(List<Map<String, Object>> chunks) {
        return chunks.stream()
                .map(
                        chunk -> {
                            Map<String, Object> copy = new LinkedHashMap<>(chunk);
                            for (String key : new ArrayList<>(copy.keySet()))
                                if (key.matches("q_[0-9]+_vec") || key.equals("vector")) {
                                    double[] v = vector(copy.remove(key));
                                    copy.put("vectorDimensions", v == null ? 0 : v.length);
                                }
                            return copy;
                        })
                .toList();
    }

    private static String hash(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        return String.format(
                "%016x", XXHashFactory.fastestJavaInstance().hash64().hash(bytes, 0, bytes.length, 0));
    }

    public static double[] vector(Object value) {
        if (value instanceof double[] v) return v;
        if (value instanceof List<?> list) {
            double[] v = new double[list.size()];
            for (int i = 0; i < v.length; i++) v[i] = ((Number) list.get(i)).doubleValue();
            return v;
        }
        return null;
    }

    private static Map<String, Object> objectMap(Map<?, ?> map) {
        Map<String, Object> copy = new LinkedHashMap<>();
        map.forEach((k, v) -> copy.put(k.toString(), v));
        return copy;
    }

    public static String required(Map<String, Object> input, String key) {
        String value = string(input, key, "");
        if (value.isBlank()) throw new IllegalArgumentException(key + " 不能为空");
        return value;
    }

    private static String string(Map<String, Object> map, String key, String fallback) {
        return map.get(key) == null ? fallback : map.get(key).toString();
    }

    private static int integer(Map<String, Object> map, String key, int fallback) {
        if (map.get(key) == null) return fallback;
        if (!(map.get(key) instanceof Number n)
                || !Double.isFinite(n.doubleValue())
                || n.doubleValue() != Math.rint(n.doubleValue())
                || n.doubleValue() < Integer.MIN_VALUE
                || n.doubleValue() > Integer.MAX_VALUE) throw new IllegalArgumentException(key + " 必须为整数");
        return n.intValue();
    }

    private static double number(Map<String, Object> map, String key, double fallback) {
        if (map.get(key) == null) return fallback;
        if (!(map.get(key) instanceof Number n) || !Double.isFinite(n.doubleValue()))
            throw new IllegalArgumentException(key + " 必须为有限数值");
        return n.doubleValue();
    }

    private static boolean bool(Map<String, Object> map, String key, boolean fallback) {
        return map.get(key) instanceof Boolean b ? b : fallback;
    }

    private static List<String> strings(Map<String, Object> map, String key, List<String> fallback) {
        if (!(map.get(key) instanceof List<?> list)) return fallback;
        return list.stream().map(Object::toString).toList();
    }
}
