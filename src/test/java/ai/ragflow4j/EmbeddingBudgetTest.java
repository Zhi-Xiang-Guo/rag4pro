package ai.ragflow4j;

import static org.junit.jupiter.api.Assertions.*;

import ai.ragflow4j.ragflow.ContextBuilder;
import ai.ragflow4j.ragflow.QwenBpe;
import com.fasterxml.jackson.databind.JsonNode;
import com.sun.net.httpserver.HttpServer;

import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class EmbeddingBudgetTest {
    private static final ContextBuilder.TokenCounter CHARACTERS =
            new ContextBuilder.TokenCounter() {
                public int count(String text) {
                    return text.length();
                }

                public String trim(String text, int limit) {
                    return text.substring(0, Math.min(text.length(), limit));
                }
            };
    private HttpServer server;
    private URI base;
    private final List<JsonNode> requests = new ArrayList<>();
    private java.util.function.Function<JsonNode, Integer> status = request -> 200;
    private int promptTokens;
    private String errorBody = "{\"error\":{\"message\":\"input length is too long\"}}";

    @BeforeEach
    void start() throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/embeddings",
                exchange -> {
                    JsonNode request = Json.MAPPER.readTree(exchange.getRequestBody());
                    requests.add(request);
                    int code = status.apply(request);
                    String body = errorBody;
                    if (code == 200) {
                        var data = new ArrayList<>();
                        for (int i = 0; i < request.path("input").size(); i++) {
                            String text = request.path("input").get(i).asText();
                            data.add(Map.of("index", i, "embedding", List.of((double) text.length(), 1.0)));
                        }
                        body =
                                Json.MAPPER.writeValueAsString(
                                        Map.of("data", data, "usage", Map.of("prompt_tokens", promptTokens)));
                    }
                    byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
                    exchange.sendResponseHeaders(code, bytes.length);
                    exchange.getResponseBody().write(bytes);
                    exchange.close();
                });
        server.start();
        base = URI.create("http://127.0.0.1:" + server.getAddress().getPort());
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    @Test
    void queryIsRawAndDefaultQwenBudgetUsesNativeModelTokens() throws Exception {
        var client = new EmbeddingClient(base, "test-key", "qwen3-embedding-4b");
        client.query("How does RAG work?");
        assertEquals("How does RAG work?", requests.getFirst().path("input").get(0).asText());
        assertFalse(requests.getFirst().has("encoding_format"));
        assertEquals(32112, client.tokenLimit());
        String longText = "9".repeat(40000);
        client.embed(List.of(longText));
        String sent = requests.getLast().path("input").get(0).asText();
        assertEquals(32112, new QwenBpe().count(sent));
        assertEquals("9".repeat(32112), sent);
        client.query(longText);
        assertEquals(sent, requests.getLast().path("input").get(0).asText());
        assertTrue(client.identity().contains("OpenAI|"));
        assertTrue(client.identity().contains("qwen3-embedding-4b"));
        assertFalse(client.identity().contains("test-key"));
        assertFalse(client.identity().contains("Instruct:"));
    }

    @Test
    void ingestionUsageCalibratesUnknownModelAcrossMatchingProviderInstances() throws Exception {
        var client = new EmbeddingClient(base, "same-key", "uncatalogued-model");
        assertEquals(1911, client.tokenLimit());
        promptTokens = 2;
        client.embedForIngestion(List.of("a"));
        assertEquals(992, client.tokenLimit());
        assertEquals(992, new EmbeddingClient(base, "same-key", "uncatalogued-model").tokenLimit());
        assertEquals(1911, new EmbeddingClient(base, "other-key", "uncatalogued-model").tokenLimit());
    }

    @Test
    void modelBoundBatchSizeMatchesSixteenInputRuntimeLimit() throws Exception {
        var client = new EmbeddingClient(base, "key", "qwen3-embedding-4b");
        List<String> texts = java.util.stream.IntStream.range(0, 33).mapToObj(i -> "text" + i).toList();
        assertEquals(33, client.embed(texts).size());
        assertEquals(
                List.of(16, 16, 1), requests.stream().map(body -> body.path("input").size()).toList());
    }

    @Test
    void shrinkLadderThenIsolatesEachInputWithoutShorteningHealthyNeighbors() throws Exception {
        var client = new EmbeddingClient(base, "key", "test-model", CHARACTERS, 1024, 16);
        status =
                request -> {
                    var input = request.path("input");
                    if (input.size() > 1) return 400;
                    String text = input.get(0).asText();
                    return text.startsWith("x") && text.length() > 64 ? 400 : 200;
                };
        List<double[]> vectors = client.embed(List.of("h".repeat(100), "x".repeat(1200)));
        assertEquals(
                List.of(992, 744, 496, 248, 124),
                requests.subList(0, 5).stream()
                        .map(body -> body.path("input").get(1).asText().length())
                        .toList());
        assertEquals(
                List.of(100, 992, 744, 496, 248, 124, 64),
                requests.subList(5, requests.size()).stream()
                        .map(body -> body.path("input").get(0).asText().length())
                        .toList());
        assertArrayEquals(new double[]{100, 1}, vectors.get(0));
        assertArrayEquals(new double[]{64, 1}, vectors.get(1));
    }

    @Test
    void isolationFailureAtFloorReturnsNoPartialResult() {
        var client = new EmbeddingClient(base, "key", "test-model", CHARACTERS, 1024, 16);
        status = request -> 400;
        var failure =
                assertThrows(
                        EmbeddingClient.UpstreamException.class, () -> client.embed(List.of("x".repeat(1200))));
        assertTrue(failure.getMessage().contains("input 0"));
        assertTrue(failure.getMessage().contains("64 tokens"));
        assertEquals(11, requests.size());
        assertEquals(64, requests.getLast().path("input").get(0).asText().length());
    }

    @Test
    void genuineProviderFailuresDoNotTriggerInputTruncationRetries() {
        var client = new EmbeddingClient(base, "key", "test-model", CHARACTERS, 1024, 16);
        status = request -> 503;
        errorBody = "input (500 tokens) is too large to process";
        assertThrows(EmbeddingClient.UpstreamException.class, () -> client.embed(List.of("long text")));
        assertEquals(1, requests.size());
    }

    @Test
    void overLimitClassificationUsesOriginalMarkersAndDelimitedCodes() {
        assertTrue(EmbeddingClient.isOverLimit(400, "maximum context length"));
        assertTrue(EmbeddingClient.isOverLimit(422, "{\"code\":20015}"));
        assertFalse(EmbeddingClient.isOverLimit(400, "{\"code\":120015}"));
        assertFalse(EmbeddingClient.isOverLimit(400, "{\"code\":200150}"));
        assertFalse(EmbeddingClient.isOverLimit(429, "token limit"));
        assertFalse(EmbeddingClient.isOverLimit(500, "too long"));
        assertTrue(EmbeddingClient.isOverLimit(500, "input is too large to process"));
        assertFalse(EmbeddingClient.isOverLimit(503, "input (500 tokens) is too large to process"));
    }

    @Test
    void ladderHasStrictlyDecreasingBudgetsAndFloorOnlyDuringIsolation() {
        assertEquals(
                List.of(32112, 24084, 16056, 8028, 4014), EmbeddingClient.overLimitLadder(32112, false));
        assertEquals(
                List.of(32112, 24084, 16056, 8028, 4014, 64), EmbeddingClient.overLimitLadder(32112, true));
        assertEquals(List.of(100, 75, 64), EmbeddingClient.overLimitLadder(100, true));
        assertEquals(List.of(32), EmbeddingClient.overLimitLadder(32, true));
        assertEquals(List.of(0), EmbeddingClient.overLimitLadder(0, true));
    }
}
