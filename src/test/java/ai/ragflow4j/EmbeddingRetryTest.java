package ai.ragflow4j;

import static org.junit.jupiter.api.Assertions.*;

import ai.ragflow4j.ragflow.ContextBuilder;
import com.sun.net.httpserver.HttpServer;

import java.net.InetSocketAddress;
import java.net.URI;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class EmbeddingRetryTest {
    private static final ContextBuilder.TokenCounter CHARACTERS =
            new ContextBuilder.TokenCounter() {
                public int count(String text) {
                    return text.length();
                }

                public String trim(String text, int limit) {
                    return text.substring(0, Math.min(text.length(), limit));
                }
            };

    private static final class Clock implements EmbeddingClient.RetryRuntime {
        long now = 1;
        boolean interruptNextSleep;
        final List<Duration> delays = new ArrayList<>();

        public long nanoTime() {
            return now;
        }

        public double random() {
            return .5;
        }

        public void sleep(Duration delay) throws InterruptedException {
            if (interruptNextSleep) {
                interruptNextSleep = false;
                throw new InterruptedException("test cancellation");
            }
            delays.add(delay);
            now += delay.toNanos();
        }
    }

    private HttpServer server;
    private URI base;
    private final AtomicInteger attempts = new AtomicInteger();
    private java.util.function.IntUnaryOperator status = attempt -> 429;
    private String body = "too many requests";

    @BeforeEach
    void start() throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/embeddings",
                exchange -> {
                    exchange.getRequestBody().readAllBytes();
                    int code = status.applyAsInt(attempts.incrementAndGet());
                    byte[] bytes =
                            (code == 200 ? "{\"data\":[{\"index\":0,\"embedding\":[1,0]}]}" : body).getBytes();
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

    private EmbeddingClient client(String key, Clock clock) {
        return new EmbeddingClient(base, key, "model", CHARACTERS, 1024, 16, clock);
    }

    @Test
    void ingestionRetriesSixTimesWithSourceBackoffSchedule() {
        var clock = new Clock();
        assertThrows(
                EmbeddingClient.UpstreamException.class,
                () -> client("key", clock).embedForIngestion(List.of("text")));
        assertEquals(6, attempts.get());
        assertEquals(
                List.of(5L, 10L, 20L, 40L, 80L), clock.delays.stream().map(Duration::toSeconds).toList());
    }

    @Test
    void bodyRetryAfterHintWinsAndSuccessfulRetryReturns() throws Exception {
        var clock = new Clock();
        body = "{\"retry_after\":2.5}";
        status = attempt -> attempt < 3 ? 503 : 200;
        assertArrayEquals(
                new double[]{1, 0}, client("key", clock).embedForIngestion(List.of("text")).getFirst());
        assertEquals(3, attempts.get());
        assertEquals(List.of(Duration.ofMillis(2500), Duration.ofMillis(2500)), clock.delays);
    }

    @Test
    void quotaCooldownIsSharedAcrossInstancesAndSeparatesCredentials() throws Exception {
        var clock = new Clock();
        clock.interruptNextSleep = true;
        status = attempt -> attempt == 1 ? 429 : 200;
        assertThrows(
                InterruptedException.class,
                () -> client("same-key", clock).embedForIngestion(List.of("text")));
        client("different-key", clock).embedForIngestion(List.of("text"));
        assertEquals(List.of(), clock.delays);
        client("same-key", clock).embedForIngestion(List.of("text"));
        assertEquals(List.of(Duration.ofSeconds(5)), clock.delays);
    }

    @Test
    void permanentHttpFailureIgnoresRateLimitWords() {
        var clock = new Clock();
        status = attempt -> 401;
        body = "rate limit; retry-after: 20; secret-value";
        var failure =
                assertThrows(
                        EmbeddingClient.UpstreamException.class,
                        () -> client("secret-value", clock).embedForIngestion(List.of("text")));
        assertEquals(1, attempts.get());
        assertEquals(List.of(), clock.delays);
        assertFalse(failure.getMessage().contains("secret-value"));
    }

    @Test
    void modelOnlyQueryDoesNotUseIngestionCooldownRetries() {
        var clock = new Clock();
        assertThrows(
                EmbeddingClient.UpstreamException.class, () -> client("key", clock).query("raw question"));
        assertEquals(1, attempts.get());
        assertEquals(List.of(), clock.delays);
    }

    @Test
    void backoffMatchesJitterHintCapAndLowerBoundRules() {
        assertEquals(Duration.ofSeconds(4), EmbeddingClient.encodeBackoff(1, 0, 0));
        assertEquals(Duration.ofSeconds(6), EmbeddingClient.encodeBackoff(1, 0, 1));
        assertEquals(Duration.ofSeconds(144), EmbeddingClient.encodeBackoff(10, 0, 1));
        assertEquals(
                Duration.ofSeconds(120),
                EmbeddingClient.encodeBackoff(1, Duration.ofSeconds(999).toNanos(), .5));
        assertEquals(
                Duration.ofMillis(100),
                EmbeddingClient.encodeBackoff(1, Duration.ofMillis(100).toNanos(), .5));
        assertEquals(
                Duration.ofMillis(2500).toNanos(), EmbeddingClient.retryAfterNanos("retry-after: 2.5"));
        assertEquals(0, EmbeddingClient.retryAfterNanos("retry-after: 0"));
        assertEquals(0, EmbeddingClient.retryAfterNanos("retry-after: -5"));
    }

    @Test
    void transientTransportClassificationMatchesSourceIntent() {
        assertTrue(EmbeddingClient.retryableTransport(new java.net.SocketTimeoutException()));
        assertTrue(
                EmbeddingClient.retryableTransport(new java.net.http.HttpTimeoutException("timeout")));
        assertTrue(
                EmbeddingClient.retryableTransport(new java.net.SocketException("Connection reset")));
        assertTrue(EmbeddingClient.retryableTransport(new java.net.SocketException("Broken pipe")));
        assertFalse(
                EmbeddingClient.retryableTransport(new java.net.UnknownHostException("no such host")));
        assertFalse(
                EmbeddingClient.retryableTransport(new java.net.ConnectException("Connection refused")));
        assertFalse(EmbeddingClient.retryableTransport(new java.io.IOException("permanent")));
    }
}
