package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import ai.ragflow4j.EmbeddingClient;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.*;
import com.sun.net.httpserver.HttpServer;

import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

/**
 * Regression coverage for HTTP-to-original-pipeline integration and catalog compensation.
 */
class CorpusServiceTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static NativeTokenizer lexical;
    private HttpServer http;
    private CorpusService corpus;
    private DocumentCatalog catalog;
    private final Map<String, Map<String, Object>> stored = new LinkedHashMap<>();
    private final List<String> embeddedQueries = new ArrayList<>();
    @TempDir
    Path temp;

    @BeforeAll
    static void nativeOpen() {
        lexical = new NativeTokenizer();
    }

    @AfterAll
    static void nativeClose() {
        lexical.close();
    }

    @BeforeEach
    void setup() throws Exception {
        http = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        http.createContext(
                "/",
                exchange -> {
                    String path = exchange.getRequestURI().getPath();
                    String body =
                            new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    Object response;
                    if (path.endsWith("/embeddings")) {
                        JsonNode input = JSON.readTree(body).path("input");
                        List<Object> rows = new ArrayList<>();
                        for (int i = 0; i < input.size(); i++) {
                            embeddedQueries.add(input.get(i).asText());
                            rows.add(Map.of("index", i, "embedding", List.of(1.0, 0.0)));
                        }
                        response = Map.of("data", rows);
                    } else if (path.endsWith("/_bulk")) {
                        String[] lines = body.split("\n");
                        for (int i = 0; i < lines.length; i += 2) {
                            String id = JSON.readTree(lines[i]).path("index").path("_id").asText();
                            stored.put(id, JSON.readValue(lines[i + 1], new TypeReference<>() {
                            }));
                        }
                        response = Map.of("errors", false);
                    } else if (path.endsWith("/_delete_by_query")) {
                        stored.clear();
                        response = Map.of("deleted", 1);
                    } else if (path.endsWith("/_search")) {
                        List<Object> hits = new ArrayList<>();
                        for (var entry : stored.entrySet())
                            hits.add(
                                    Map.of(
                                            "_id",
                                            entry.getKey(),
                                            "_index",
                                            "test",
                                            "_score",
                                            0.9,
                                            "_source",
                                            entry.getValue()));
                        response = Map.of("hits", Map.of("total", Map.of("value", hits.size()), "hits", hits));
                    } else response = Map.of();
                    byte[] output = JSON.writeValueAsBytes(response);
                    exchange.sendResponseHeaders(200, output.length);
                    exchange.getResponseBody().write(output);
                    exchange.close();
                });
        http.start();
        URI base = URI.create("http://127.0.0.1:" + http.getAddress().getPort());
        catalog = new DocumentCatalog(temp.resolve("catalog.json"));
        EmbeddingClient embeddings =
                new EmbeddingClient(base, "fixture-key", "fixture", new Cl100k(), 2048, 16);
        IndexingPipeline.Embedder indexEmbedder =
                texts -> texts.stream().map(text -> new double[]{1, 0}).toList();
        corpus =
                new CorpusService(
                        new ElasticsearchStore(base, "test"),
                        catalog,
                        lexical,
                        embeddings,
                        indexEmbedder,
                        new LocalChatModel(base, "fixture", 8192),
                        8192);
    }

    @AfterEach
    void stop() {
        http.stop(0);
    }

    @Test
    void batchCatalogFailureRemovesItsNewChunksAndDoesNotPublishDocuments() throws Exception {
        Files.createDirectory(temp.resolve("catalog.json"));
        var docs=List.of(Map.of("docId","a","title","A.txt","text","Alpha policies."));
        assertThrows(Exception.class, () -> corpus.indexBatch(Map.of("datasetId","batch","documents",docs)));
        assertTrue(stored.isEmpty());
        assertTrue(catalog.list("batch").isEmpty());
    }

    @Test
    void httpTokenAndBodyProtocolAreEnforcedAndRequestIdsAreReturned() throws Exception {
        try (var api = new ai.ragflow4j.StandardServer(new InetSocketAddress("127.0.0.1",0),corpus,1,"local-test-token")) {
            api.start();
            var client=java.net.http.HttpClient.newHttpClient();
            String base="http://127.0.0.1:"+api.address().getPort();
            var unauthorized=client.send(java.net.http.HttpRequest.newBuilder(URI.create(base+"/api/documents"))
                    .GET().build(),java.net.http.HttpResponse.BodyHandlers.ofString());
            assertEquals(401,unauthorized.statusCode());
            assertTrue(unauthorized.headers().firstValue("X-Request-Id").isPresent());
            var invalid=client.send(java.net.http.HttpRequest.newBuilder(URI.create(base+"/api/retrieve"))
                    .header("Authorization","Bearer local-test-token").header("Content-Type","application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString("broken" )).build(),
                    java.net.http.HttpResponse.BodyHandlers.ofString());
            assertEquals(400,invalid.statusCode());
            var valid=client.send(java.net.http.HttpRequest.newBuilder(URI.create(base+"/api/documents"))
                    .header("Authorization","Bearer local-test-token").GET().build(),
                    java.net.http.HttpResponse.BodyHandlers.ofString());
            assertEquals(200,valid.statusCode()); // errors release the admission permit
        }
    }

    @Test
    void extractiveHttpDistinguishesExhaustedModelFailureFromMalformedSelection() throws Exception {
        var modelCalls = new java.util.concurrent.atomic.AtomicInteger();
        http.createContext("/api/chat", exchange -> {
            int count = modelCalls.incrementAndGet();
            byte[] output = (count <= 2 ? "{}" : "{\"message\":{\"content\":\"{\\\"supported\\\":true,\\\"ids\\\":[99]}\"}}")
                    .getBytes(StandardCharsets.UTF_8);
            try (exchange) {
                exchange.sendResponseHeaders(count <= 2 ? 500 : 200, output.length);
                exchange.getResponseBody().write(output);
            }
        });
        corpus.indexBatch(Map.of("datasetId", "kb", "documents",
                List.of(Map.of("docId", "a", "title", "Refund terms", "text", "Refunds take three days."))));
        try (var api = new ai.ragflow4j.StandardServer(new InetSocketAddress("127.0.0.1", 0), corpus, 1, "")) {
            api.start();
            var client = java.net.http.HttpClient.newHttpClient();
            var request = java.net.http.HttpRequest.newBuilder(
                            URI.create("http://127.0.0.1:" + api.address().getPort() + "/api/chat"))
                    .header("Content-Type", "application/json")
                    .POST(java.net.http.HttpRequest.BodyPublishers.ofString(JSON.writeValueAsString(
                            Map.of("datasetId", "kb", "answerMode", "extractive", "strategy", "lexical",
                                    "question", "How long do refunds take?"))))
                    .build();
            var unavailable = client.send(request, java.net.http.HttpResponse.BodyHandlers.ofString());
            JsonNode error = JSON.readTree(unavailable.body());
            assertEquals(503, unavailable.statusCode());
            assertEquals("local_model_unavailable", error.path("code").asText());
            assertEquals(2, error.path("attempts").asInt());
            assertFalse(error.has("answer"));

            var invalid = client.send(request, java.net.http.HttpResponse.BodyHandlers.ofString());
            assertEquals(502, invalid.statusCode());
            assertEquals("invalid_evidence_selection", JSON.readTree(invalid.body()).path("code").asText());
            assertEquals(4, modelCalls.get());
        }
    }

    @Test
    void batchPersistsOriginalFieldsAndRejectsExistingIdsBeforeWriting() throws Exception {
        var docs = List.of(Map.of("docId", "a", "title", "A.txt", "text", "Alpha policies."),
                Map.of("docId", "b", "title", "B.txt", "text", "Beta policies."));
        var result = corpus.indexBatch(Map.of("datasetId", "batch", "documents", docs));
        assertEquals(2, result.get("documentCount"));
        assertEquals(2, catalog.list("batch").size());
        assertEquals(2, stored.size());
        for (var chunk : stored.values()) {
            assertTrue(chunk.containsKey("content_ltks"));
            assertTrue(chunk.containsKey("q_2_vec"));
            assertEquals("batch", chunk.get("kb_id"));
        }
        assertThrows(IllegalArgumentException.class,
                () -> corpus.indexBatch(Map.of("datasetId", "batch", "documents", docs)));
        assertEquals(2, stored.size());
    }

    private Map<String, Object> retrievalInput() {
        return new LinkedHashMap<>(Map.of("datasetId", "dataset", "question", "alpha"));
    }

    @Test
    void languageComesFromDatasetAndMixedLanguagesUseEnglish() throws Exception {
        assertEquals("", corpus.retrievalRequest(retrievalInput(), false).language);
        catalog.put("dataset", "a", Map.of("language", "Slovak"));
        catalog.put("other", "a", Map.of("language", "Chinese"));
        assertEquals("Slovak", corpus.retrievalRequest(retrievalInput(), false).language);
        catalog.put("dataset", "b", Map.of("language", " slovak "));
        assertEquals("Slovak", corpus.datasetLanguage("dataset"));
        catalog.put("dataset", "c", Map.of("language", "Czech"));
        assertEquals("", corpus.datasetLanguage("dataset"));
        Map<String, Object> explicit = retrievalInput();
        explicit.put("language", "Chinese");
        assertEquals("Chinese", corpus.retrievalRequest(explicit, false).language);
    }

    @Test
    void optionalRerankerAndRankFeaturesReachRetrievalRequest() {
        Map<String, Object> input = retrievalInput();
        input.put("rankFeature", Map.of("subject", 2.0));
        input.put("highlight", true);
        var request = corpus.retrievalRequest(input, false);
        assertEquals(Map.of("subject", 2.0), request.rankFeature);
        assertTrue(request.highlight);
        assertNull(request.rerankModel);
        input.put("useReranker", true);
        assertThrows(IllegalArgumentException.class, () -> corpus.retrievalRequest(input, false));
        Reranker.Model model = (question, documents) -> List.of();
        corpus.configureReranker(model);
        assertSame(model, corpus.retrievalRequest(input, false).rerankModel);
    }

    @Test
    void invalidThresholdsAndCandidatePoolsFailBeforeCalls() {
        for (Object invalid : List.of(-0.1, 1.1, Double.NaN, Double.POSITIVE_INFINITY, "0.3")) {
            Map<String, Object> input = retrievalInput();
            input.put("threshold", invalid);
            assertThrows(IllegalArgumentException.class, () -> corpus.retrievalRequest(input, false));
        }
        Map<String, Object> request = retrievalInput();
        request.put("knnTopK", 0);
        assertThrows(IllegalArgumentException.class, () -> corpus.retrievalRequest(request, false));
        request.put("knnTopK", 10);
        request.put("knnNumCandidates", 9);
        assertThrows(IllegalArgumentException.class, () -> corpus.retrievalRequest(request, false));
        request.put("knnTopK", 1.2);
        assertThrows(IllegalArgumentException.class, () -> corpus.retrievalRequest(request, false));
    }

    @Test
    void duplicateTextCountsDistinctPersistedIdsAndKeepsDocumentMetadata() throws Exception {
        Map<String, Object> input =
                new LinkedHashMap<>(
                        Map.of(
                                "datasetId",
                                "dataset",
                                "docId",
                                "doc",
                                "title",
                                "test.txt",
                                "text",
                                "alpha\nalpha\n",
                                "chunkTokenSize",
                                0,
                                "metadata",
                                Map.of("author", "Alice")));
        Map<String, Object> result = corpus.index(input);
        assertEquals(1, result.get("chunkCount"));
        assertEquals(0, result.get("parentCount"));
        assertEquals(1, stored.size());
        assertEquals(1, catalog.get("dataset", "doc").get("chunkCount"));
        String documentId = DocumentCatalog.identity("dataset", "doc");
        Map<String, Object> chunk =
                new LinkedHashMap<>(Map.of("kb_id", "dataset", "doc_id", documentId));
        Map<String, Object> foreign =
                new LinkedHashMap<>(Map.of("kb_id", "other", "doc_id", documentId));
        corpus.enrichDocumentMetadata("dataset", List.of(chunk, foreign));
        assertEquals(Map.of("author", "Alice"), chunk.get("document_metadata"));
        assertFalse(foreign.containsKey("document_metadata"));
    }

    @Test
    void markdownBookkeepingIsRemovedAndDeleteCompensatesCatalogFailure() throws Exception {
        corpus.index(
                Map.of(
                        "datasetId",
                        "dataset",
                        "docId",
                        "doc",
                        "title",
                        "test.md",
                        "text",
                        "# Heading\n\nbody"));
        assertFalse(stored.isEmpty());
        for (Map<String, Object> source : stored.values()) assertFalse(source.containsKey("layout"));
        Map<String, Map<String, Object>> before = new LinkedHashMap<>(stored);
        Files.delete(temp.resolve("catalog.json"));
        Files.createDirectory(temp.resolve("catalog.json"));
        assertThrows(Exception.class, () -> corpus.delete("dataset", "doc"));
        assertEquals(before.keySet(), stored.keySet());
        assertNotNull(catalog.get("dataset", "doc"));
    }

    @Test
    void noRefinementConcatenatesOnlyLatestThreeUserQuestions() throws Exception {
        var input =
                new LinkedHashMap<String, Object>(
                        Map.of(
                                "datasetId",
                                "dataset",
                                "refineMultiturn",
                                false,
                                "messages",
                                List.of(
                                        Map.of("role", "user", "content", "first"),
                                        Map.of("role", "assistant", "content", "reply"),
                                        Map.of("role", "user", "content", "second"),
                                        Map.of("role", "user", "content", "third"),
                                        Map.of("role", "user", "content", "fourth"))));
        Map<String, Object> response = corpus.answer(input);
        assertEquals("second third fourth", response.get("query"));
        assertEquals(List.of("second third fourth"), embeddedQueries);
        assertEquals(
                "Sorry! No relevant content was found in the knowledge base!", response.get("answer"));
    }
}
