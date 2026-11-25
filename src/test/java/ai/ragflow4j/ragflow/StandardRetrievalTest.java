// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
// http://www.apache.org/licenses/LICENSE-2.0
package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.*;
import com.sun.net.httpserver.HttpServer;

import java.net.*;
import java.nio.file.*;
import java.util.*;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

/**
 * 原 retrieval_test.go 窗口、回退和父子块契约，经真实 ES HTTP 适配器验证。
 */
class StandardRetrievalTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static NativeTokenizer tokenizer;
    private static QueryBuilder query;
    private HttpServer server;
    private final List<JsonNode> requests = new ArrayList<>();
    private final Deque<Object> responses = new ArrayDeque<>();
    private ElasticsearchStore store;
    @TempDir
    static Path temp;

    @BeforeAll
    static void openNative() throws Exception {
        tokenizer = new NativeTokenizer();
        Path emptySynonyms = temp.resolve("synonym.json");
        Files.writeString(emptySynonyms, "{}");
        query =
                new QueryBuilder(
                        tokenizer, new TermWeightDealer(tokenizer), new Synonym(emptySynonyms, null));
    }

    @AfterAll
    static void closeNative() {
        tokenizer.close();
    }

    @BeforeEach
    void server() throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/",
                exchange -> {
                    byte[] raw = exchange.getRequestBody().readAllBytes();
                    requests.add(raw.length == 0 ? JSON.createObjectNode() : JSON.readTree(raw));
                    Object response = responses.removeFirst();
                    int status = response instanceof Integer value ? value : 200;
                    byte[] body =
                            JSON.writeValueAsBytes(
                                    response instanceof Integer ? Map.of("error", "test failure") : response);
                    exchange.sendResponseHeaders(status, body.length);
                    exchange.getResponseBody().write(body);
                    exchange.close();
                });
        server.start();
        store =
                new ElasticsearchStore(
                        URI.create("http://127.0.0.1:" + server.getAddress().getPort()), "ragflow_test");
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    @Test
    void trueLexicalBaselineDoesNotCallEmbeddingOrKnn() throws Exception {
        responses.add(hits(List.of(hit("lex", 1, source("alpha")))));
        var retrieval = new StandardRetrieval(store, query, text -> { throw new AssertionError("lexical must not embed"); });
        var request = request(); request.strategy = "lexical";
        var result = retrieval.retrieve(request);
        assertEquals("lex",result.chunks().getFirst().get("chunk_id"));
        assertFalse(requests.getFirst().has("knn"));
        assertEquals(1,requests.size());
    }

    @Test
    void rrfIncludesDenseCandidateExcludedByLexicalBranchAndRetainsScope() throws Exception {
        responses.add(hits(List.of(hit("lex", 1, source("alpha")))));
        responses.add(hits(List.of(hit("dense", .9, source("semantic paraphrase")))));
        var request = request(); request.strategy = "rrf"; request.pageSize = 2;
        var result = service().retrieve(request);
        assertEquals(Set.of("lex","dense"),result.chunks().stream().map(c -> c.get("chunk_id")).collect(java.util.stream.Collectors.toSet()));
        assertFalse(requests.getFirst().has("knn"));
        assertTrue(requests.getLast().has("knn"));
        assertFalse(requests.getLast().toString().contains("query_string"));
        assertTrue(requests.getLast().toString().contains("kb-1"));
    }

    private StandardRetrieval service() {
        return new StandardRetrieval(store, query, text -> new double[]{1, 0});
    }

    private StandardRetrieval.Request request() {
        var request = new StandardRetrieval.Request();
        request.datasetId = "kb-1";
        request.question = "alpha";
        request.existingDocumentIds = Set.of("doc-1");
        return request;
    }

    private static Map<String, Object> hit(String id, double score, Map<String, Object> source) {
        return Map.of("_id", id, "_index", "ragflow_test", "_score", score, "_source", source);
    }

    private static Map<String, Object> source(String text) {
        return Map.of(
                "doc_id",
                "doc-1",
                "kb_id",
                "kb-1",
                "docnm_kwd",
                "document",
                "content_ltks",
                text,
                "content_with_weight",
                text,
                "q_2_vec",
                List.of(1.0, 0.0));
    }

    private static Map<String, Object> hits(List<Map<String, Object>> hits) {
        return Map.of("hits", Map.of("total", Map.of("value", hits.size()), "hits", hits));
    }

    @Test
    void originalDefaultAndWindowValidation() {
        var request = request();
        StandardRetrieval.defaultsAndValidate(request);
        assertEquals(64, request.rerankCandidatesCount);
        assertEquals(1024, request.knnTopK);
        assertEquals(2048, request.knnNumCandidates);
        assertEquals(0.2, request.similarityThreshold);
        assertEquals(0.3, request.vectorSimilarityWeight);
        assertEquals(Map.of("pagerank_fea", 10.0), request.rankFeature);
        request.page = 7;
        request.pageSize = 10;
        assertThrows(
                IllegalArgumentException.class, () -> StandardRetrieval.defaultsAndValidate(request));
        request.page = 1;
        request.rerankCandidatesCount = 1025;
        assertThrows(
                IllegalArgumentException.class, () -> StandardRetrieval.defaultsAndValidate(request));
        assertTrue(requests.isEmpty());
    }

    @Test
    void stablePaginationCountsAllThresholdValidCandidates() throws Exception {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int i = 0; i < 64; i++)
            rows.add(hit(String.format("chunk-%02d", i), 0.9, source("alpha")));
        responses.add(hits(rows));
        var request = request();
        request.vectorOnly = true;
        request.page = 6;
        request.pageSize = 10;
        var result = service().retrieve(request);
        assertEquals(64, result.total());
        assertEquals("chunk-50", result.chunks().getFirst().get("chunk_id"));
        assertEquals("chunk-59", result.chunks().getLast().get("chunk_id"));
        assertEquals(64, result.docAggs().getFirst().get("count"));
        assertEquals(1, requests.size());
        assertEquals(64, requests.getFirst().path("size").asInt());
        assertFalse(requests.getFirst().has("query"));
        assertEquals(0.9, result.chunks().getFirst().get("vector_similarity"));
        assertEquals(0.0, result.chunks().getFirst().get("term_similarity"));
        assertArrayEquals(new double[2], (double[]) result.chunks().getFirst().get("vector"));
    }

    @Test
    void zeroOnlyFallbackKeepsScopeAndExactFusionWeight() throws Exception {
        responses.add(hits(List.of()));
        responses.add(hits(List.of()));
        responses.add(hits(List.of(hit("c1", 0.9, source("alpha")))));
        responses.add(hits(List.of(hit("c1", 0.8, Map.of()))));
        var request = request();
        request.knnTopK = 7;
        request.knnNumCandidates = 19;
        request.docIds = List.of("doc-1");
        request.filter = Map.of("category_kwd", "allowed");
        var result = service().retrieve(request);
        assertEquals(1, result.total());
        assertEquals(4, requests.size());
        assertEquals(0, requests.get(0).path("query").path("bool").path("boost").asDouble());
        assertEquals(0.2, requests.get(0).path("knn").path("similarity").asDouble());
        assertEquals(0.17, requests.get(1).path("knn").path("similarity").asDouble());
        assertEquals(0.17, requests.get(2).path("knn").path("similarity").asDouble());
        assertFalse(requests.get(2).has("query"));
        assertEquals(7, requests.get(2).path("knn").path("k").asInt());
        assertEquals(19, requests.get(2).path("knn").path("num_candidates").asInt());
        assertTrue(requests.get(2).path("knn").path("filter").toString().contains("allowed"));
        double expectedToken = (0.4 + 1e-9) / (0.7 + 1e-9);
        assertEquals(expectedToken, (Double) result.chunks().getFirst().get("term_similarity"), 1e-12);
        assertEquals(
                0.7 * expectedToken + 0.3 * 0.8,
                (Double) result.chunks().getFirst().get("similarity"),
                1e-12);
    }

    @Test
    void disabledDenseFallbackAndWeakHitDoNotBroadenSearch() throws Exception {
        responses.add(hits(List.of()));
        responses.add(hits(List.of()));
        var request = request();
        request.allowDenseFallback = false;
        assertEquals(0, service().retrieve(request).total());
        assertEquals(2, requests.size());
    }

    @Test
    void knnFailureFallsBackToRawVectorsAndPrunesGhosts() throws Exception {
        Map<String, Object> ghost = new LinkedHashMap<>(source("alpha"));
        ghost.put("doc_id", "deleted-document");
        responses.add(hits(List.of(hit("c1", 0.5, source("alpha")), hit("ghost", 0.9, ghost))));
        responses.add(503);
        var result = service().retrieve(request());
        assertEquals(1, result.total());
        assertEquals("c1", result.chunks().getFirst().get("chunk_id"));
        assertEquals(1.0, result.chunks().getFirst().get("vector_similarity"));
        assertTrue(
                result.trace().stream()
                        .anyMatch(entry -> entry.get("stage").equals("local_rerank_fallback")));
    }

    @Test
    void parentExpansionAveragesChildrenAndChecksDocumentIdentity() {
        Map<String, Object> parent = new LinkedHashMap<>(source("complete parent text"));
        parent.put("available_int", 0);
        responses.add(hits(List.of(hit("parent", 0, parent))));
        List<Map<String, Object>> children =
                List.of(
                        Map.of(
                                "chunk_id",
                                "a",
                                "mom_id",
                                "parent",
                                "doc_id",
                                "doc-1",
                                "kb_id",
                                "kb-1",
                                "content_ltks",
                                "first",
                                "similarity",
                                0.8,
                                "important_kwd",
                                List.of("one")),
                        Map.of(
                                "chunk_id",
                                "b",
                                "mom_id",
                                "parent",
                                "doc_id",
                                "doc-1",
                                "kb_id",
                                "kb-1",
                                "content_ltks",
                                "second",
                                "similarity",
                                0.4,
                                "important_kwd",
                                new String[]{"ignored"}));
        List<Map<String, Object>> expanded = service().expandParents(children);
        assertEquals(1, expanded.size());
        assertEquals(0.6, (Double) expanded.getFirst().get("similarity"), 1e-12);
        assertEquals("first second", expanded.getFirst().get("content_ltks"));
        assertEquals(List.of("one"), expanded.getFirst().get("important_kwd"));
        assertEquals(1024, ((double[]) expanded.getFirst().get("vector")).length);
        Map<String, Object> unrelated = new LinkedHashMap<>(parent);
        unrelated.put("doc_id", "other-document");
        responses.add(hits(List.of(hit("parent", 0, unrelated))));
        assertEquals(children, service().expandParents(children));
    }
}
