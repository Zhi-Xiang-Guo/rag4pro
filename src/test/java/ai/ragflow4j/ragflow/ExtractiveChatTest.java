package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class ExtractiveChatTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String SOURCE = "  退款审核后 3 个工作日内到账。超时请提交工单。";
    private static final String REFUSAL = "资料不足，无法回答。";
    private static final ChatPipeline.Retrieval SOURCES = new ChatPipeline.Retrieval(
            List.of(Map.of("chunk_id", "chunk-a", "doc_id", "doc-a", "docnm_kwd", "退款规则", "content_with_weight", SOURCE)),
            List.of());

    @Test
    void rendersOnlySelectedOriginalSpansAndUsesSourceOffsets() throws Exception {
        var app = new ExtractiveChat((messages, schema, generation, deadline) -> {
            var payload = JSON.readTree(messages.getLast().content());
            assertEquals("超时怎么办？", payload.path("questions").get(1).asText());
            assertEquals("超时请提交工单。", payload.path("candidateExcerpts").get(1).get(2).asText());
            return model("{\"answer\":\"draft\",\"ids\":[1]}");
        }, new Cl100k(), 8192);
        var answer = app.complete(List.of("退款多久？", "超时怎么办？"), SOURCES, REFUSAL, true);
        assertEquals("超时请提交工单。 [ID:0]", answer.answer());
        var reference = JSON.valueToTree(answer.reference());
        var span = reference.path("spans").get(0);
        assertEquals("chunk-a", reference.path("chunks").get(0).path("id").asText());
        assertEquals(span.path("text").asText(), SOURCE.substring(span.path("start").asInt(), span.path("end").asInt()));
        assertEquals(List.of(0), answer.citedIndices());
    }

    @Test
    void noPassagesSkipsModelAndEmptySelectionDoesNotTreatRetrievedTextAsCitations() throws Exception {
        var calls = new AtomicInteger();
        var app = new ExtractiveChat((messages, schema, generation, deadline) -> {
            calls.incrementAndGet();
            return model("{\"answer\":\"INSUFFICIENT\",\"ids\":[]}");
        }, new Cl100k(), 8192);
        var empty = app.complete(List.of("退款手续费？"), new ChatPipeline.Retrieval(List.of(), List.of()), REFUSAL, true);
        assertEquals(0, calls.get());
        assertEquals(REFUSAL, empty.answer());
        var unsupported = app.complete(List.of("退款手续费？"), SOURCES, REFUSAL, true);
        assertEquals(1, calls.get());
        assertTrue(unsupported.emptyResponse());
        assertEquals(REFUSAL, unsupported.answer());
        assertEquals(List.of(), unsupported.reference().get("chunks"));
    }

    @Test
    void malformedOutOfRangeOrDuplicateIdsCannotProduceAPartialAnswer() {
        for (String invalid : List.of("{\"answer\":\"draft\",\"ids\":[0,99]}", "{\"answer\":\"draft\",\"ids\":[0,0]}",
                "{\"answer\":\"draft\",\"ids\":[\"0\"]}", "{\"answer\":\"draft\",\"ids\":[0]} {\"ids\":[]}",
                "{\"answer\":\"draft\",\"ids\":[0],\"unexpected\":\"free text\"}",
                "{\"answer\":\"INSUFFICIENT\",\"ids\":[0]}", "{\"answer\":\"draft\",\"ids\":[]}")) {
            var app = new ExtractiveChat((messages, schema, generation, deadline) -> model(invalid), new Cl100k(), 8192);
            assertThrows(LocalChatModel.InvalidSelection.class,
                    () -> app.complete(List.of("退款多久？"), SOURCES, REFUSAL, true), invalid);
        }
    }

    @Test
    void correctsInconsistentEvidenceWithTheSameDeadlineAndSources() throws Exception {
        var calls = new AtomicInteger();
        var firstDeadline = new java.util.concurrent.atomic.AtomicLong();
        var app = new ExtractiveChat((messages, schema, generation, deadline) -> {
            if (calls.incrementAndGet() == 1) {
                firstDeadline.set(deadline);
                return model("{\"answer\":\"INSUFFICIENT\",\"ids\":[0]}");
            }
            assertEquals(firstDeadline.get(), deadline);
            assertTrue(messages.getFirst().content().contains("did not satisfy the schema"));
            assertEquals(List.of("ids"), schema.get("required"));
            return model("{\"ids\":[0]}");
        }, new Cl100k(), 8192);

        var result = app.complete(List.of("退款审核后多久到账？"), SOURCES, REFUSAL, true);
        assertEquals("退款审核后 3 个工作日内到账。 [ID:0]", result.answer());
        assertEquals(2, result.audit().get("attempts"));
        assertEquals(true, result.audit().get("selectionRepaired"));
    }

    @Test
    void retryableOllamaErrorRetriesOnceAndRepeatedFailureIsDistinguishableFromRefusal() throws Exception {
        var calls = new AtomicInteger();
        var api = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        api.createContext("/api/chat", exchange -> {
            int call = calls.incrementAndGet();
            byte[] payload = (call == 2 ? "{\"message\":{\"content\":\"{\\\"answer\\\":\\\"draft\\\",\\\"ids\\\":[0]}\"}}" : "{}")
                    .getBytes(StandardCharsets.UTF_8);
            try (exchange) {
                exchange.sendResponseHeaders(call == 2 ? 200 : 500, payload.length);
                exchange.getResponseBody().write(payload);
            }
        });
        api.start();
        try {
            var client = new LocalChatModel(URI.create("http://127.0.0.1:" + api.getAddress().getPort()), "local-test", 8192);
            var app = new ExtractiveChat(client::selectEvidence, new Cl100k(), client.contextLength());
            var result = app.complete(List.of("退款多久？"), SOURCES, REFUSAL, true);
            assertTrue(result.answer().contains("3 个工作日"));
            assertEquals(2, result.audit().get("attempts"));
            var failure = assertThrows(LocalChatModel.Unavailable.class,
                    () -> app.complete(List.of("退款多久？"), SOURCES, REFUSAL, true));
            assertEquals(2, failure.attempts());
            assertEquals(4, calls.get());
        } finally {
            api.stop(0);
        }
    }

    @Test
    void openAiChatAndEvidenceSelectionUseTheSameAuthenticatedEndpoint() throws Exception {
        var request = new AtomicReference<JsonNode>();
        var authorization = new AtomicReference<String>();
        var api = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        api.createContext("/gateway/chat/completions", exchange -> {
            try (exchange) {
                request.set(JSON.readTree(exchange.getRequestBody()));
                authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
                byte[] payload = JSON.writeValueAsBytes(Map.of(
                        "choices", List.of(Map.of("message", Map.of(
                                "content", "{\"answer\":\"draft\",\"ids\":[0]}",
                                "reasoning_content", "Checked the source"))),
                        "usage", Map.of("total_tokens", 42)));
                exchange.sendResponseHeaders(200, payload.length);
                exchange.getResponseBody().write(payload);
            }
        });
        api.start();
        try {
            URI base = URI.create("http://127.0.0.1:" + api.getAddress().getPort() + "/gateway/");
            var client = LocalChatModel.openAi(base, "test-key", "remote-test", 8192);
            var reply = client.complete(List.of(new ContextBuilder.Message("user", "退款多久？")),
                    Map.of("max_tokens", 128, "stream", true, "num_ctx", 4096));
            assertEquals("Bearer test-key", authorization.get());
            assertEquals("remote-test", request.get().path("model").asText());
            assertEquals(128, request.get().path("max_tokens").asInt());
            assertFalse(request.get().path("stream").asBoolean());
            assertFalse(request.get().has("num_ctx"));
            assertEquals("Checked the source", reply.reasoning());
            assertEquals(42, reply.usage().get("total_tokens"));

            var app = new ExtractiveChat(client::selectEvidence, new Cl100k(), client.contextLength());
            var result = app.complete(List.of("退款多久？"), SOURCES, REFUSAL, true);
            assertTrue(result.answer().contains("3 个工作日"));
            assertEquals(192, request.get().path("max_tokens").asInt());
            assertEquals("json_schema", request.get().path("response_format").path("type").asText());
            assertTrue(request.get().at("/response_format/json_schema/schema/properties/ids").isObject());
            assertFalse(request.get().has("options"));
        } finally {
            api.stop(0);
        }
    }

    private static LocalChatModel.Selection model(String json) {
        return new LocalChatModel.Selection(new ChatPipeline.ModelAnswer(json), 1);
    }
}
