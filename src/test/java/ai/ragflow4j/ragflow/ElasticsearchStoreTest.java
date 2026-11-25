package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;

import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class ElasticsearchStoreTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private HttpServer server;
    private ElasticsearchStore store;
    private final List<JsonNode> requests = new ArrayList<>();
    private final List<String> paths = new ArrayList<>();
    private final List<String> bulk = new ArrayList<>();
    private String response = "{\"hits\":{\"total\":{\"value\":0},\"hits\":[]}}";

    @BeforeEach
    void start() throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/",
                exchange -> {
                    paths.add(exchange.getRequestURI().toString());
                    String body =
                            new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    String reply = response;
                    if (exchange.getRequestURI().getPath().endsWith("/_bulk")) {
                        bulk.add(body);
                        assertEquals(
                                "application/x-ndjson", exchange.getRequestHeaders().getFirst("Content-Type"));
                        reply = "{\"errors\":false}";
                    } else if (!body.isEmpty()) requests.add(JSON.readTree(body));
                    byte[] bytes = reply.getBytes(StandardCharsets.UTF_8);
                    exchange.sendResponseHeaders(200, bytes.length);
                    exchange.getResponseBody().write(bytes);
                    exchange.close();
                });
        server.start();
        store =
                new ElasticsearchStore(
                        URI.create("http://127.0.0.1:" + server.getAddress().getPort()), "chunks");
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    @Test
    void hybridMatchesActiveGoFusionAndImmutableKnnFilter() throws Exception {
        var text =
                new QueryBuilder.MatchTextExpr(
                        List.of(),
                        "Virus OR \"NEW AGE\" AND NOT Ransom",
                        5,
                        Map.of("minimum_should_match", .285));
        store.search(
                "kb",
                List.of("doc"),
                text,
                new double[]{.3, .4},
                20000,
                5,
                3,
                .2,
                1,
                Map.of("id", List.of("c1", "c2")),
                Map.of("pagerank_fea", 2.0, "biology", 3.0),
                List.of("content_with_weight", "q_2_vec"));
        JsonNode body = requests.getLast();
        JsonNode bool = body.path("query").path("bool");
        assertEquals(0, bool.path("boost").asDouble());
        assertEquals(
                "virus OR \"new age\" AND NOT ransom",
                bool.path("must").get(0).path("query_string").path("query").asText());
        assertEquals(
                "29%", bool.path("must").get(0).path("query_string").path("minimum_should_match").asText());
        assertEquals(7, bool.path("must").get(0).path("query_string").path("fields").size());
        assertEquals(
                "tag_feas.biology", bool.path("should").get(0).path("rank_feature").path("field").asText());
        assertEquals(
                "pagerank_fea", bool.path("should").get(1).path("rank_feature").path("field").asText());
        assertFalse(body.path("knn").has("boost"));
        assertFalse(body.path("knn").path("filter").path("bool").has("should"));
        assertEquals(10000, body.path("knn").path("k").asInt());
        assertEquals(10000, body.path("knn").path("num_candidates").asInt());
        assertEquals(
                JSON.readTree("[\"content_with_weight\",\"q_2_vec\",\"pagerank_fea\",\"tag_feas\"]"),
                body.path("_source"));
        assertEquals(JSON.readTree("[\"q_2_vec\"]"), body.path("fields"));
        assertTrue(paths.getLast().contains("allow_partial_search_results=false"));
    }

    @Test
    void denseOnlyHasNoScoredQueryAndAvailabilityIncludesMissingField() throws Exception {
        store.search(
                "kb",
                List.of(),
                null,
                new double[]{1, 0},
                0,
                0,
                0,
                0,
                1,
                Map.of(),
                Map.of("pagerank_fea", 1.0));
        JsonNode body = requests.getLast();
        assertFalse(body.has("query"));
        assertEquals(30, body.path("size").asInt());
        assertEquals(30, body.path("knn").path("k").asInt());
        assertEquals(30, body.path("knn").path("num_candidates").asInt());
        assertEquals(
                JSON.readTree("{\"bool\":{\"must_not\":[{\"range\":{\"available_int\":{\"lt\":1}}}]}}"),
                body.path("knn").path("filter").path("bool").path("filter").get(1));
    }

    @Test
    void filtersHonorIdAlternativesBooleanGroupsAndExclusions() throws Exception {
        Map<String, Object> filter = new LinkedHashMap<>();
        filter.put("available_int", 0);
        filter.put("id", "chunk");
        filter.put("or", List.of(Map.of("docnm_kwd", "A"), Map.of("exists", "tag_feas")));
        filter.put("must_not", Map.of("exists", "deleted_kwd"));
        store.search("kb", List.of(), null, null, 0, 0, 10, 0, .5, filter, Map.of("pagerank_fea", 1.0));
        JsonNode bool = requests.getLast().path("query").path("bool");
        assertEquals(
                JSON.readTree("[{\"exists\":{\"field\":\"deleted_kwd\"}}]"), bool.path("must_not"));
        assertEquals(
                JSON.readTree("{\"range\":{\"available_int\":{\"lt\":1}}}"), bool.path("filter").get(1));
        JsonNode idFilter = bool.path("filter").get(2).path("bool");
        assertEquals(1, idFilter.path("minimum_should_match").asInt());
        assertEquals("chunk", idFilter.path("should").get(1).path("term").path("_id").asText());
        assertFalse(bool.has("minimum_should_match"));
        assertFalse(bool.has("boost"));
    }

    @Test
    void knnScoresUseCandidateIdentityAndPreserveEngineScores() throws Exception {
        response = "{\"hits\":{\"hits\":[{\"_id\":\"one\",\"_score\":0.75}]}}";
        var candidate = Map.<String, Object>of("id", "wrong", "_id", "one", "_index", "other_index");
        assertEquals(
                Map.of("one", .75), store.knnScores(List.of(candidate, candidate), new double[]{1, 0}));
        assertTrue(paths.getLast().startsWith("/other_index/_search"));
        assertEquals(1, requests.getLast().path("knn").path("k").asInt());
        assertEquals(
                JSON.readTree("{\"terms\":{\"id\":[\"one\"]}}"),
                requests.getLast().path("knn").path("filter"));
        assertThrows(
                IllegalArgumentException.class,
                () -> store.knnScores(List.of(Map.of("_id", "x")), new double[]{1}));
    }

    @Test
    void getFieldsAndHighlightMatchGoHelpers() {
        Map<String, Object> chunk =
                Map.of(
                        "_id",
                        "x",
                        "tag_feas",
                        Map.of("test", 2),
                        "docnm_kwd",
                        List.of("A"),
                        "position_int",
                        List.of(List.of(1, 2, 3, 4, 5)),
                        "available_int",
                        1,
                        "pagerank_fea",
                        2.0,
                        "content_with_weight",
                        "Learning learns C++ 学习机器学习");
        var projected =
                store
                        .getFields(
                                List.of(chunk),
                                List.of(
                                        "id", "tag_feas", "docnm_kwd", "position_int", "available_int", "pagerank_fea"))
                        .get("x");
        assertEquals("x", projected.get("id"));
        assertEquals("{\"test\":2}", projected.get("tag_feas"));
        assertEquals("A", projected.get("docnm_kwd"));
        assertEquals(List.of(List.of(1, 2, 3, 4, 5)), projected.get("position_int"));
        assertEquals(1, projected.get("available_int"));
        assertEquals("2", projected.get("pagerank_fea"));
        assertEquals(
                "<em>Learning</em> <em>learns</em> <em>C++</em> <em>学习</em><em>机器学习</em>",
                store
                        .getHighlight(
                                List.of(chunk), List.of("learn", "学习", "机器学习", "C++"), "content_with_weight")
                        .get("x"));
    }

    @Test
    void parentLookupUsesAllThreeScopesAndIncludesUnavailableRows() throws Exception {
        response =
                "{\"hits\":{\"hits\":[{\"_id\":\"physical\",\"_index\":\"chunks\",\"_source\":{\"id\":\"parent\",\"available_int\":0,\"content_with_weight\":\"Parent\"}}]}}";
        assertEquals("Parent", store.getParent("parent", "doc", "kb").get("content_with_weight"));
        JsonNode body = requests.getLast();
        assertEquals(1, body.path("size").asInt());
        assertFalse(body.toString().contains("available_int"));
        assertTrue(body.toString().contains("\"doc_id\":\"doc\""));
        assertTrue(body.toString().contains("\"kb_id\":\"kb\""));
        assertTrue(body.toString().contains("\"_id\":\"parent\""));
    }

    @Test
    void bulkStampsDatasetAvailabilityAndRoundsOrderedRankFeatures() throws Exception {
        List<Map<String, Object>> chunks = new ArrayList<>();
        for (int i = 0; i < 501; i++) {
            chunks.add(
                    Map.of(
                            "id",
                            "c" + i,
                            "doc_id",
                            "d",
                            "kb_id",
                            "wrong",
                            "tag_feas",
                            Map.of("a", 2.5, "b", 4.4, "c", 2.5)));
        }
        store.replace("kb", "d", chunks);
        assertEquals(2, bulk.size());
        assertEquals(1000, bulk.getFirst().lines().count());
        JsonNode source = JSON.readTree(bulk.getLast().split("\n")[1]);
        assertEquals("kb", source.path("kb_id").asText());
        assertEquals(1, source.path("available_int").asInt());
        assertEquals("{\"b\":4,\"a\":3,\"c\":3}", source.path("tag_feas").toString());
    }
}
