package ai.ragflow4j;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.JsonNode;

import java.net.URI;
import java.net.http.*;
import java.util.*;

import org.junit.jupiter.api.Test;

/**
 * 手动执行真实标准链路，验证 HTTP / 模型 / ES / 引用边界，测试后清理自身数据。
 */
class StandardLiveIT {
    @Test
    void standardIndexRetrieveAndLocalAnswer() throws Exception {
        String base = System.getProperty("ragflow4j.url", "http://127.0.0.1:18080");
        String dataset = "verify-standard-" + UUID.randomUUID();
        HttpClient http = HttpClient.newHttpClient();
        try {
            JsonNode indexed =
                    post(
                            http,
                            base + "/api/documents",
                            Map.of(
                                    "datasetId",
                                    dataset,
                                    "docId",
                                    "refund",
                                    "title",
                                    "退款政策.txt",
                                    "text",
                                    "退款在审核通过后的三个工作日内原路返回支付账户。",
                                    "language",
                                    "Chinese"));
            assertEquals(1, indexed.path("chunkCount").asInt());
            assertEquals(2560, indexed.path("chunks").get(0).path("vectorDimensions").asInt());
            assertTrue(indexed.path("chunks").get(0).path("content_ltks").asText().contains("退款"));
            JsonNode result =
                    post(
                            http,
                            base + "/api/retrieve",
                            Map.of("datasetId", dataset, "question", "退款多久到账", "threshold", 0.1));
            assertFalse(result.path("chunks").isEmpty());
            assertTrue(
                    result.path("chunks").get(0).path("content_with_weight").asText().contains("三个工作日"));
            assertTrue(result.path("chunks").get(0).has("term_similarity"));
            JsonNode answer =
                    post(
                            http,
                            base + "/api/chat",
                            Map.of(
                                    "datasetId",
                                    dataset,
                                    "question",
                                    "退款多久到账",
                                    "threshold",
                                    0.1,
                                    "generation",
                                    Map.of("temperature", 0, "max_tokens", 256)));
            assertFalse(answer.path("answer").asText().isBlank());
            assertFalse(answer.path("reference").path("chunks").isEmpty());
            assertTrue(answer.path("prompt").asText().contains("三个工作日"));
            assertEquals("qwen2.5:0.5b", answer.path("model").asText());
            assertTrue(answer.path("usage").path("total_tokens").asInt() > 0);
            var selected = post(http, base + "/api/chat", Map.of("datasetId", dataset,
                    "answerMode", "extractive", "question", "退款多久到账？", "evaluationTrace", true));
            assertEquals("answered", selected.path("answerStatus").asText());
            assertTrue(selected.path("answer").asText().contains("三个工作日"));
            var evidence = selected.path("reference").path("spans").get(0);
            var source = selected.path("reference").path("chunks").get(evidence.path("citationIndex").asInt());
            assertEquals(evidence.path("text").asText(), source.path("content").asText()
                    .substring(evidence.path("start").asInt(), evidence.path("end").asInt()));
            var unsupported = post(http, base + "/api/chat", Map.of("datasetId", dataset,
                    "answerMode", "extractive", "question", "退款手续费是多少？", "evaluationTrace", true));
            assertEquals("insufficient_evidence", unsupported.path("answerStatus").asText());
            assertTrue(unsupported.path("reference").path("chunks").isEmpty());
            assertTrue(unsupported.path("evaluation").path("modelBoundary").path("modelInvoked").asBoolean(),
                    "Test must exercise refusal with a related but insufficient document");
        } finally {
            var response =
                    http.send(
                            HttpRequest.newBuilder(
                                            URI.create(base + "/api/documents?datasetId=" + dataset + "&docId=refund"))
                                    .DELETE()
                                    .build(),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(200, response.statusCode());
        }
    }

    private JsonNode post(HttpClient http, String url, Object value) throws Exception {
        var request =
                HttpRequest.newBuilder(URI.create(url))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(Json.MAPPER.writeValueAsString(value)))
                        .build();
        var response = http.send(request, HttpResponse.BodyHandlers.ofString());
        assertEquals(200, response.statusCode(), response.body());
        return Json.MAPPER.readTree(response.body());
    }
}
