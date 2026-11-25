package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.*;

/**
 * 聊天模型 HTTP 客户端，负责 Ollama 原生协议与 OpenAI 兼容协议的请求和响应转换。
 */
public final class LocalChatModel implements ChatPipeline.ChatModel {
    static final Duration SELECTION_TIMEOUT = Duration.ofSeconds(30);
    private static final Duration RETRY_DELAY = Duration.ofMillis(200);
    private static final Set<String> OPENAI_GENERATION = Set.of(
            "temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "seed", "stop");

    public record Selection(ChatPipeline.ModelAnswer response, int attempts) {}

    public static final class Unavailable extends IOException {
        private final int attempts;

        Unavailable(int status, int attempts) {
            super(status == 0 ? "聊天模型连接超时或断开" : "聊天模型 HTTP " + status);
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
    private final boolean openAi;
    private final String apiKey;

    public LocalChatModel(URI base, String model, int contextLength) {
        this(base, model, contextLength, false, "");
    }

    public static LocalChatModel openAi(URI base, String apiKey, String model, int contextLength) {
        String host = Objects.toString(base.getHost(), "");
        boolean loopback = "http".equals(base.getScheme())
                && Set.of("localhost", "127.0.0.1", "::1", "[::1]").contains(host);
        if ((!loopback && !"https".equals(base.getScheme())) || host.isEmpty()
                || base.getUserInfo() != null || base.getRawQuery() != null || base.getRawFragment() != null)
            throw new IllegalArgumentException("EVIDENCE_RAG_CHAT_URL 需要有效的 HTTPS 基础地址");
        if (apiKey.isBlank()) throw new IllegalArgumentException("EVIDENCE_RAG_CHAT_API_KEY is required");
        if (model.isBlank()) throw new IllegalArgumentException("EVIDENCE_RAG_CHAT_MODEL is required");
        return new LocalChatModel(base, model, contextLength, true, apiKey.strip());
    }

    private LocalChatModel(URI base, String model, int contextLength, boolean openAi, String apiKey) {
        String path = openAi ? "/chat/completions" : "/api/chat";
        this.endpoint = URI.create(base.toString().replaceAll("/+$", "") + path);
        this.model = model;
        this.contextLength = contextLength;
        this.openAi = openAi;
        this.apiKey = apiKey;
    }

    @Override
    public ChatPipeline.ModelAnswer complete(
            List<ContextBuilder.Message> messages, Map<String, Object> generation) throws Exception {
        return complete(messages, generation, null, Duration.ofMinutes(5));
    }

    ChatPipeline.ModelAnswer complete(
            List<ContextBuilder.Message> messages, Map<String, Object> generation,
            Map<String, Object> schema, Duration timeout) throws Exception {
        var response = send(payload(messages, generation, schema), timeout);
        if (response.statusCode() != 200)
            throw new IllegalStateException("聊天模型 HTTP " + response.statusCode());
        return decode(response.body());
    }

    /** 选择证据只需很短的 JSON。瞬时失败最多重试一次，两次共用 30 秒预算。 */
    public Selection selectEvidence(List<ContextBuilder.Message> messages, Map<String, Object> schema,
                                    Map<String, Object> generation) throws IOException, InterruptedException {
        return selectEvidence(messages, schema, generation, System.nanoTime() + SELECTION_TIMEOUT.toNanos());
    }

    Selection selectEvidence(List<ContextBuilder.Message> messages, Map<String, Object> schema,
                             Map<String, Object> generation, long deadline) throws IOException, InterruptedException {
        String payload = payload(messages, generation, schema);
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
                    throw new InvalidSelection("聊天模型返回无效响应", attempts);
                }
            }
            boolean retryable = status == 0 || status == 408 || status == 429 || status >= 500;
            if (!retryable || attempts == 2 || deadline - System.nanoTime() <= RETRY_DELAY.toNanos()) break;
            Thread.sleep(RETRY_DELAY);
        }
        throw new Unavailable(status, attempts);
    }

    private String payload(List<ContextBuilder.Message> messages, Map<String, Object> generation,
                           Map<String, Object> schema) throws IOException {
        Map<String, Object> body = new LinkedHashMap<>(Map.of("model", model, "messages", messages, "stream", false));
        if (openAi) {
            generation.forEach((key, value) -> {
                String parameter = key.equals("num_predict") ? "max_tokens" : key;
                if (OPENAI_GENERATION.contains(parameter)) body.put(parameter, value);
            });
            if (schema != null) {
                body.put("response_format", Map.of("type", "json_schema",
                        "json_schema", Map.of("name", "evidence_selection", "strict", true, "schema", schema)));
            }
        } else {
            Map<String, Object> options = new LinkedHashMap<>();
            if (schema == null) {
                options.put("num_ctx", contextLength);
                generation.forEach((key, value) -> options.put(key.equals("max_tokens") ? "num_predict" : key, value));
            } else {
                options.putAll(generation);
                options.put("num_ctx", contextLength);
                body.put("think", false);
                body.put("format", schema);
            }
            body.put("options", options);
        }
        return json.writeValueAsString(body);
    }

    private HttpResponse<String> send(String payload, Duration timeout) throws IOException, InterruptedException {
        var request = HttpRequest.newBuilder(endpoint)
                .timeout(timeout)
                .header("Content-Type", "application/json");
        if (openAi) request.header("Authorization", "Bearer " + apiKey);
        return client.send(request.POST(HttpRequest.BodyPublishers.ofString(payload)).build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private ChatPipeline.ModelAnswer decode(String payload) throws IOException {
        var result = json.readTree(payload);
        if (result == null || !result.isObject()) throw new IllegalStateException("聊天模型返回无效响应");
        if (result.hasNonNull("error")) throw new IllegalStateException("聊天模型请求失败");
        var message = openAi ? result.path("choices").path(0).path("message") : result.path("message");
        String answer = message.path("content").asText("");
        String reasoning = message.path(openAi ? "reasoning_content" : "thinking").asText("");
        if (answer.isBlank() && reasoning.isBlank()) throw new IllegalStateException("聊天模型返回空回答");
        if (openAi) {
            var usage = result.path("usage");
            return new ChatPipeline.ModelAnswer(answer, reasoning,
                    usage.isObject() ? json.convertValue(usage, new TypeReference<>() {}) : Map.of());
        }
        int prompt = result.path("prompt_eval_count").asInt(),
                completion = result.path("eval_count").asInt();
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
