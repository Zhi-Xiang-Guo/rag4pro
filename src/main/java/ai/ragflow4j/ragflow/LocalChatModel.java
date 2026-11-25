package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.*;

/**
 * 本机 Ollama 原生协议适配。与 RAG 编排独立，消息、生成参数和模型 usage 原样传递。
 */
public final class LocalChatModel implements ChatPipeline.ChatModel {
    static final Duration SELECTION_TIMEOUT = Duration.ofSeconds(30);
    private static final Duration RETRY_DELAY = Duration.ofMillis(200);

    public record Selection(ChatPipeline.ModelAnswer response, int attempts) {}

    public static final class Unavailable extends IOException {
        private final int attempts;

        Unavailable(int status, int attempts) {
            super(status == 0 ? "本地模型连接超时或断开" : "本地聊天模型 HTTP " + status);
            this.attempts = attempts;
        }

        public int attempts() {
            return attempts;
        }
    }

    public static final class InvalidSelection extends IOException {
        private final int attempts;

        InvalidSelection(String message, int attempts) {
            super(message);
            this.attempts = attempts;
        }

        public int attempts() {
            return attempts;
        }
    }

    private final HttpClient client =
            HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    private final ObjectMapper json = new ObjectMapper();
    private final URI endpoint;
    private final String model;
    private final int contextLength;

    public LocalChatModel(URI base, String model, int contextLength) {
        this.endpoint = URI.create(base.toString().replaceAll("/+$", "") + "/api/chat");
        this.model = model;
        this.contextLength = contextLength;
    }

    @Override
    public ChatPipeline.ModelAnswer complete(
            List<ContextBuilder.Message> messages, Map<String, Object> generation) throws Exception {
        Map<String, Object> options = new LinkedHashMap<>();
        options.put("num_ctx", contextLength);
        for (var entry : generation.entrySet()) {
            String key =
                    switch (entry.getKey()) {
                        case "max_tokens" -> "num_predict";
                        default -> entry.getKey();
                    };
            options.put(key, entry.getValue());
        }
        Map<String, Object> body =
                Map.of("model", model, "messages", messages, "stream", false, "options", options);
        var response = send(json.writeValueAsString(body), Duration.ofMinutes(5));
        if (response.statusCode() != 200)
            throw new IllegalStateException("本地聊天模型 HTTP " + response.statusCode());
        return decode(response.body());
    }

    /** 选择证据只需很短的 JSON。瞬时失败最多重试一次，两次共用 30 秒预算。 */
    public Selection selectEvidence(List<ContextBuilder.Message> messages, Map<String, Object> schema,
                                    Map<String, Object> generation) throws IOException, InterruptedException {
        return selectEvidence(messages, schema, generation, System.nanoTime() + SELECTION_TIMEOUT.toNanos());
    }

    Selection selectEvidence(List<ContextBuilder.Message> messages, Map<String, Object> schema,
                             Map<String, Object> generation, long deadline) throws IOException, InterruptedException {
        Map<String, Object> options = new LinkedHashMap<>(generation);
        options.put("num_ctx", contextLength);
        String payload = json.writeValueAsString(Map.of("model", model, "messages", messages, "stream", false,
                "think", false, "format", schema, "options", options));
        int status = 0;
        int attempts = 0;
        while (attempts < 2) {
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0) break;
            attempts++;
            HttpResponse<String> response;
            try {
                // Let an in-progress inference use the remaining budget. Cutting it off early wastes
                // the completed work and can leave the retry with too little time to finish either.
                response = send(payload, Duration.ofNanos(remaining));
                status = response.statusCode();
            } catch (IOException failure) {
                response = null;
                status = 0;
            }
            if (status == 200) {
                try {
                    return new Selection(decode(response.body()), attempts);
                } catch (IOException | IllegalStateException invalid) {
                    throw new InvalidSelection("本地模型返回无效响应", attempts);
                }
            }
            boolean retryable = status == 0 || status == 408 || status == 429 || status >= 500;
            if (!retryable || attempts == 2 || deadline - System.nanoTime() <= RETRY_DELAY.toNanos()) break;
            Thread.sleep(RETRY_DELAY);
        }
        throw new Unavailable(status, attempts);
    }

    private HttpResponse<String> send(String payload, Duration timeout) throws IOException, InterruptedException {
        var request = HttpRequest.newBuilder(endpoint)
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(payload))
                .build();
        return client.send(request, HttpResponse.BodyHandlers.ofString());
    }

    private ChatPipeline.ModelAnswer decode(String payload) throws IOException {
        var result = json.readTree(payload);
        if (result == null || !result.isObject()) throw new IllegalStateException("本地模型返回无效响应");
        if (result.has("error")) throw new IllegalStateException("本地模型请求失败");
        int prompt = result.path("prompt_eval_count").asInt(),
                completion = result.path("eval_count").asInt();
        String answer = result.path("message").path("content").asText();
        String reasoning = result.path("message").path("thinking").asText();
        if (answer.isBlank() && reasoning.isBlank()) throw new IllegalStateException("本地模型返回空回答");
        Map<String, Object> usage =
                prompt == 0 && completion == 0
                        ? Map.of()
                        : Map.of(
                        "prompt_tokens",
                        prompt,
                        "completion_tokens",
                        completion,
                        "total_tokens",
                        prompt + completion);
        return new ChatPipeline.ModelAnswer(answer, reasoning, usage);
    }

    public String model() {
        return model;
    }

    public int contextLength() {
        return contextLength;
    }
}
