package ai.ragflow4j;

import ai.ragflow4j.ragflow.Cl100k;
import ai.ragflow4j.ragflow.ContextBuilder;
import ai.ragflow4j.ragflow.IndexingPipeline;
import ai.ragflow4j.ragflow.QwenBpe;

import java.net.URI;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/**
 * embedding 网关适配器：把文本发送到 OpenAI 兼容协议端点，并按输入顺序还原向量。
 *
 * <p>对应 OpenAIModel.Embed 和 embed_limit.go 的预算/重试路径；建索引与查询共用它， 回答模型的调用另见
 * LocalChatModel。密钥只用于请求认证，不应出现在返回值与日志中。
 */
public final class EmbeddingClient implements Embeddings {
    private static final int DEFAULT_BATCH_SIZE = 16;
    private static final int OVER_LIMIT_FLOOR = 64;
    private static final Pattern OVER_LIMIT_STATUS = Pattern.compile("\\b(?:400|413|422)\\b");
    private static final Pattern OVER_LIMIT_CODE = Pattern.compile("(?<![0-9])20015(?![0-9])");
    private static final List<String> OVER_LIMIT_MARKERS =
            List.of(
                    "too long",
                    "too many tokens",
                    "maximum context",
                    "context length",
                    "context_length",
                    "input length",
                    "token limit",
                    "reduce the length",
                    "maximum allowed",
                    "max context size");
    private static final Pattern RETRY_AFTER =
            Pattern.compile("(?i)retry[-_ ]?after[\"'\\s:=]+(\\d+(\\.\\d+)?)");
    private static final ConcurrentHashMap<String, Long> COOLDOWNS = new ConcurrentHashMap<>();

    interface RetryRuntime {
        long nanoTime();

        void sleep(Duration delay) throws InterruptedException;

        double random();
    }

    private static final RetryRuntime SYSTEM_RETRY =
            new RetryRuntime() {
                public long nanoTime() {
                    return System.nanoTime();
                }

                public void sleep(Duration delay) throws InterruptedException {
                    Thread.sleep(delay);
                }

                public double random() {
                    return java.util.concurrent.ThreadLocalRandom.current().nextDouble();
                }
            };
    private static final ConcurrentHashMap<String, Double> CALIBRATION = new ConcurrentHashMap<>();

    private static final class QwenCounter {
        private static final QwenBpe INSTANCE = new QwenBpe();
    }

    private static final class FallbackCounter {
        private static final Cl100k INSTANCE = new Cl100k();
    }

    private final HttpClient http =
            HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15)).build();
    private final URI endpoint;
    private final String key;
    private final String model;
    private final ContextBuilder.TokenCounter counter;
    private final int maxTokens;
    private final int batchSize;
    private final boolean exactCounter;
    private final String calibrationKey;
    private final RetryRuntime retryRuntime;

    public EmbeddingClient(URI base, String key, String model) {
        this(base, key, model, isQwenEmbedding(model) ? 32768 : 2048);
    }

    EmbeddingClient(URI base, String key, String model, int maxTokens) {
        this(
                base,
                key,
                model,
                isQwenEmbedding(model) ? QwenCounter.INSTANCE : FallbackCounter.INSTANCE,
                maxTokens,
                DEFAULT_BATCH_SIZE,
                isQwenEmbedding(model),
                SYSTEM_RETRY);
    }

    /**
     * Explicit model-bound counter/window, corresponding to the resolved model catalog entry.
     */
    public EmbeddingClient(
            URI base,
            String key,
            String model,
            ContextBuilder.TokenCounter counter,
            int maxTokens,
            int batchSize) {
        this(base, key, model, counter, maxTokens, batchSize, true, SYSTEM_RETRY);
    }

    EmbeddingClient(
            URI base,
            String key,
            String model,
            ContextBuilder.TokenCounter counter,
            int maxTokens,
            int batchSize,
            RetryRuntime retryRuntime) {
        this(base, key, model, counter, maxTokens, batchSize, true, retryRuntime);
    }

    private EmbeddingClient(
            URI base,
            String key,
            String model,
            ContextBuilder.TokenCounter counter,
            int maxTokens,
            int batchSize,
            boolean exactCounter,
            RetryRuntime retryRuntime) {
        boolean https = "https".equals(base.getScheme());
        String host = Objects.toString(base.getHost(), "");
        String token = Objects.toString(key, "").strip();
        boolean localHttp = "http".equals(base.getScheme())
                && (Set.of("localhost", "127.0.0.1", "::1", "[::1]").contains(host)
                || (host.equals("host.docker.internal") && token.isEmpty()));
        if ((!https && !localHttp) || host.isEmpty() || base.getUserInfo() != null
                || base.getRawQuery() != null || base.getRawFragment() != null) {
            throw new IllegalArgumentException("远程模型服务需要 HTTPS，Ollama 可使用本机 HTTP");
        }
        this.endpoint = URI.create(base.toString().replaceAll("/+$", "") + "/embeddings");
        this.key = https ? required(token, "EVIDENCE_RAG_API_KEY") : token;
        this.model = required(model, "EVIDENCE_RAG_MODEL");
        this.counter = Objects.requireNonNull(counter, "counter");
        this.maxTokens = maxTokens > 0 ? maxTokens : 2048;
        if (batchSize <= 0) throw new IllegalArgumentException("Embedding batch size must be positive");
        this.batchSize = batchSize;
        this.exactCounter = exactCounter;
        this.retryRuntime = Objects.requireNonNull(retryRuntime, "retryRuntime");
        try {
            calibrationKey =
                    endpoint
                            + "|"
                            + model
                            + "|"
                            + HexFormat.of()
                            .formatHex(
                                    MessageDigest.getInstance("SHA-256")
                                            .digest(this.key.getBytes(StandardCharsets.UTF_8)));
        } catch (java.security.NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(impossible);
        }
    }

    private static String required(String value, String field) {
        if (value == null || value.isBlank())
            throw new IllegalArgumentException(field + " is required");
        return value.strip();
    }

    static boolean isQwenEmbedding(String model) {
        if (model == null) return false;
        String name = model.toLowerCase(Locale.ROOT);
        return name.equals("qwen3-embedding-4b") || name.equals("qwen/qwen3-embedding-4b");
    }

    @Override
    public String identity() {
        return "OpenAI|"
                + endpoint
                + "|"
                + model
                + "|raw-input|"
                + counter.getClass().getName()
                + "|"
                + maxTokens;
    }

    public int maxTokens() {
        return maxTokens;
    }

    public int batchSize() {
        return batchSize;
    }

    public int tokenLimit() {
        double ratio = exactCounter ? 1 : Math.max(1, CALIBRATION.getOrDefault(calibrationKey, 1.05));
        int calibratedWindow = Math.max(1, (int) Math.floor(maxTokens / ratio));
        return IndexingPipeline.embeddingTokenLimit(calibratedWindow);
    }

    public String trim(String text) {
        return counter.trim(text, tokenLimit());
    }

    private record Embedded(List<double[]> vectors, int inputTokens) {
    }

    @Override
    public List<double[]> embed(List<String> texts) throws Exception {
        return embedResult(texts).vectors();
    }

    private Embedded embedResult(List<String> texts) throws Exception {
        Objects.requireNonNull(texts, "texts");
        if (texts.isEmpty()) return new Embedded(List.of(), 0);
        List<double[]> vectors = new ArrayList<>();
        int inputTokens = 0;
        for (int start = 0; start < texts.size(); start += batchSize) {
            checkInterrupted();
            Embedded batch = embedBatch(texts.subList(start, Math.min(start + batchSize, texts.size())));
            vectors.addAll(batch.vectors());
            inputTokens += batch.inputTokens();
        }
        int dimension = vectors.getFirst().length;
        if (vectors.stream().anyMatch(vector -> vector.length != dimension)) {
            throw new UpstreamException("Embedding 返回的向量维度不一致");
        }
        return new Embedded(List.copyOf(vectors), inputTokens);
    }

    /**
     * Ingestion Encode adds its provider-scoped cooldown around the model wrapper.
     */
    public List<double[]> embedForIngestion(List<String> texts) throws Exception {
        if (texts.isEmpty()) return List.of();
        UpstreamException lastFailure = null;
        for (int budget : overLimitLadder(tokenLimit(), false)) {
            List<String> candidate = trimAll(texts, budget);
            try {
                Embedded result = embedWithRetry(candidate);
                observeUsage(candidate, result.inputTokens());
                return result.vectors();
            } catch (UpstreamException failure) {
                if (!failure.overLimit) throw failure;
                observeOverLimit(candidate.stream().mapToInt(counter::count).max().orElse(0));
                lastFailure = failure;
            }
        }
        try {
            List<double[]> vectors = new ArrayList<>();
            for (int index = 0; index < texts.size(); index++) {
                boolean fitted = false;
                for (int budget : overLimitLadder(tokenLimit(), true)) {
                    checkInterrupted();
                    String candidate = counter.trim(texts.get(index), budget);
                    try {
                        Embedded result = embedWithRetry(List.of(candidate));
                        observeUsage(List.of(candidate), result.inputTokens());
                        vectors.add(result.vectors().getFirst());
                        fitted = true;
                        break;
                    } catch (UpstreamException failure) {
                        if (!failure.overLimit) throw failure;
                        observeOverLimit(counter.count(candidate));
                    }
                }
                if (!fitted)
                    throw new UpstreamException("Embedding input " + index + " does not fit at 64 tokens");
            }
            return vectors;
        } catch (UpstreamException failure) {
            // Encode returns its last batch over-limit rejection if isolation also failed.
            if (lastFailure != null) throw lastFailure;
            throw failure;
        }
    }

    private Embedded embedWithRetry(List<String> texts) throws Exception {
        for (int attempt = 1; attempt <= 6; attempt++) {
            checkInterrupted();
            Long deadline = COOLDOWNS.get(calibrationKey);
            if (deadline != null) {
                long remaining = deadline - retryRuntime.nanoTime();
                if (remaining > 0) retryRuntime.sleep(Duration.ofNanos(remaining));
            }
            try {
                return embedResult(texts);
            } catch (UpstreamException failure) {
                if (!failure.retryable || attempt == 6) throw failure;
                Duration delay =
                        encodeBackoff(
                                attempt,
                                failure.retryAfterNanos,
                                failure.retryAfterNanos > 0 ? .5 : retryRuntime.random());
                startCooldown(delay);
                retryRuntime.sleep(delay);
            }
        }
        throw new IllegalStateException("Unreachable embedding retry state");
    }

    private void startCooldown(Duration delay) {
        if (delay.isZero() || delay.isNegative()) return;
        long now = retryRuntime.nanoTime();
        COOLDOWNS.merge(calibrationKey, now + delay.toNanos(), Math::max);
        if (COOLDOWNS.size() > 64) COOLDOWNS.entrySet().removeIf(entry -> entry.getValue() - now <= 0);
    }

    static Duration encodeBackoff(int attempt, long retryAfterNanos, double random) {
        if (retryAfterNanos > 0)
            return Duration.ofNanos(Math.min(retryAfterNanos, Duration.ofSeconds(120).toNanos()));
        double seconds = Math.min(5 * Math.pow(2, attempt - 1), 120);
        seconds += (random * 2 - 1) * seconds * .2;
        return Duration.ofNanos((long) (Math.max(1, seconds) * 1_000_000_000));
    }

    private Embedded embedBatch(List<String> texts) throws Exception {
        int window = tokenLimit();
        for (int budget : overLimitLadder(window, false)) {
            checkInterrupted();
            try {
                return send(trimAll(texts, budget));
            } catch (UpstreamException failure) {
                if (!failure.overLimit) throw failure;
                observeOverLimit(texts.stream().mapToInt(counter::count).max().orElse(0));
            }
        }
        List<double[]> result = new ArrayList<>();
        int lastInputTokens = 0;
        for (int index = 0; index < texts.size(); index++) {
            String text = texts.get(index);
            boolean fitted = false;
            UpstreamException isolationFailure = null;
            for (int budget : overLimitLadder(window, true)) {
                checkInterrupted();
                String trimmed = counter.trim(text, budget);
                try {
                    Embedded single = send(List.of(trimmed));
                    result.add(single.vectors().getFirst());
                    // embedIsolating reuses the model usage receiver for each single input.
                    lastInputTokens = single.inputTokens();
                    fitted = true;
                    break;
                } catch (UpstreamException failure) {
                    if (!failure.overLimit) throw failure;
                    observeOverLimit(counter.count(trimmed));
                    isolationFailure = failure;
                }
            }
            if (!fitted) {
                throw new UpstreamException(
                        "Embedding input " + index + " does not fit the model window even at 64 tokens",
                        isolationFailure);
            }
        }
        return new Embedded(result, lastInputTokens);
    }

    private List<String> trimAll(List<String> texts, int budget) {
        return texts.stream().map(text -> counter.trim(text, budget)).toList();
    }

    private void observeOverLimit(int ownCount) {
        if (!exactCounter && ownCount > 0) {
            double impliedRatio = (double) maxTokens / ownCount * 1.01;
            CALIBRATION.merge(calibrationKey, Math.max(1.05, impliedRatio), Math::max);
        }
    }

    private void observeUsage(List<String> texts, int reportedTokens) {
        if (exactCounter || reportedTokens <= 0) return;
        int ownCount = texts.stream().mapToInt(counter::count).sum();
        if (ownCount > 0)
            CALIBRATION.merge(
                    calibrationKey, Math.max(1.05, (double) reportedTokens / ownCount), Math::max);
    }

    static List<Integer> overLimitLadder(int window, boolean includeFloor) {
        if (window <= 0) return List.of(0);
        List<Integer> limits = new ArrayList<>(List.of(window));
        for (double factor : new double[]{.75, .5, .25, .125}) {
            int next = Math.max(OVER_LIMIT_FLOOR, (int) (window * factor));
            if (next < limits.getLast()) limits.add(next);
        }
        if (includeFloor && limits.getLast() > OVER_LIMIT_FLOOR) limits.add(OVER_LIMIT_FLOOR);
        return limits;
    }

    private static void checkInterrupted() throws InterruptedException {
        if (Thread.currentThread().isInterrupted())
            throw new InterruptedException("Embedding call interrupted");
    }

    private Embedded send(List<String> batch) throws Exception {
        String body = Json.MAPPER.writeValueAsString(Map.of("model", model, "input", batch));
        var request = HttpRequest.newBuilder(endpoint)
                .timeout(Duration.ofSeconds(90))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body));
        if (!key.isEmpty()) request.header("Authorization", "Bearer " + key);
        HttpResponse<String> response;
        try {
            response = http.send(request.build(), HttpResponse.BodyHandlers.ofString());
        } catch (java.io.IOException failure) {
            throw new UpstreamException("Embedding 网关连接失败或超时", false, retryableTransport(failure), 0);
        }
        if (response.statusCode() != 200) {
            int status = response.statusCode();
            throw new UpstreamException(
                    "Embedding 网关返回 HTTP " + status,
                    isOverLimit(status, response.body()),
                    status == 429 || status >= 500,
                    retryAfterNanos(response.body()));
        }
        return parseVectors(response.body(), batch.size());
    }

    static boolean isOverLimit(int status, String responseBody) {
        String message =
                ("OpenAI embeddings API error: " + status + ", body: " + responseBody)
                        .toLowerCase(Locale.ROOT);
        if (!OVER_LIMIT_STATUS.matcher(message).find()) {
            return status == 500 && message.contains("is too large to process");
        }
        return OVER_LIMIT_MARKERS.stream().anyMatch(message::contains)
                || OVER_LIMIT_CODE.matcher(message).find();
    }

    private Embedded parseVectors(String body, int count) {
        try {
            var root = Json.MAPPER.readTree(body);
            var data = root.path("data");
            if (!data.isArray() || data.size() != count) throw new IllegalArgumentException();
            double[][] ordered = new double[count][];
            for (var item : data) {
                if (!item.path("index").isIntegralNumber() || !item.path("index").canConvertToInt())
                    throw new IllegalArgumentException();
                int index = item.path("index").intValue();
                var embedding = item.path("embedding");
                if (index < 0
                        || index >= ordered.length
                        || ordered[index] != null
                        || !embedding.isArray()
                        || embedding.isEmpty()) {
                    throw new IllegalArgumentException();
                }
                double[] vector = new double[embedding.size()];
                double norm = 0;
                for (int i = 0; i < vector.length; i++) {
                    if (!embedding.get(i).isNumber()) throw new IllegalArgumentException();
                    vector[i] = embedding.get(i).doubleValue();
                    if (!Double.isFinite(vector[i])) throw new IllegalArgumentException();
                    norm += vector[i] * vector[i];
                }
                if (!Double.isFinite(norm) || norm <= 0) throw new IllegalArgumentException();
                ordered[index] = vector;
            }
            int dimension = ordered[0].length;
            if (Arrays.stream(ordered).anyMatch(vector -> vector.length != dimension))
                throw new IllegalArgumentException();
            return new Embedded(Arrays.asList(ordered), root.path("usage").path("prompt_tokens").asInt());
        } catch (Exception failure) {
            throw new UpstreamException("Embedding 响应格式无效：向量数量、index 或数值不合法");
        }
    }

    static long retryAfterNanos(String errorText) {
        var matcher = RETRY_AFTER.matcher(errorText);
        if (!matcher.find()) return 0;
        try {
            double seconds = Double.parseDouble(matcher.group(1));
            if (!Double.isFinite(seconds) || seconds <= 0) return 0;
            return (long) (Math.min(seconds, 120) * 1_000_000_000);
        } catch (NumberFormatException ignored) {
            return 0;
        }
    }

    static boolean retryableTransport(Throwable error) {
        for (Throwable current = error; current != null; current = current.getCause()) {
            if (current instanceof java.net.http.HttpTimeoutException
                    || current instanceof java.net.SocketTimeoutException) return true;
            if (current instanceof java.net.UnknownHostException
                    || current instanceof java.net.ConnectException) return false;
            String message = Objects.toString(current.getMessage(), "").toLowerCase(Locale.ROOT);
            if (current instanceof java.net.SocketException
                    && (message.contains("connection reset")
                    || message.contains("broken pipe")
                    || message.contains("connection abort"))) return true;
            if (message.contains("http2: server sent goaway and closed the connection")
                    && message.contains("errcode=no_error")) return true;
        }
        return false;
    }

    public static final class UpstreamException extends RuntimeException {
        private final boolean overLimit;
        private final boolean retryable;
        private final long retryAfterNanos;

        public UpstreamException(String message) {
            this(message, false, false, 0);
        }

        private UpstreamException(String message, UpstreamException original) {
            this(
                    message,
                    original != null && original.overLimit,
                    original != null && original.retryable,
                    original == null ? 0 : original.retryAfterNanos);
        }

        private UpstreamException(
                String message, boolean overLimit, boolean retryable, long retryAfterNanos) {
            super(message);
            this.overLimit = overLimit;
            this.retryable = retryable;
            this.retryAfterNanos = retryAfterNanos;
        }
    }
}
