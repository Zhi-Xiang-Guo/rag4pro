package ai.ragflow4j;

import static org.junit.jupiter.api.Assertions.*;

import com.sun.net.httpserver.HttpServer;

import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.junit.jupiter.api.Test;

class EmbeddingClientTest {
    @Test
    void embedsWithLocalOllamaWithoutSendingAnAuthorizationHeader() throws Exception {
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/embeddings", exchange -> {
            exchange.getRequestBody().readAllBytes();
            int status = exchange.getRequestHeaders().containsKey("Authorization") ? 400 : 200;
            byte[] payload = "{\"data\":[{\"index\":0,\"embedding\":[1,0]}]}".getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(status, payload.length);
            exchange.getResponseBody().write(payload);
            exchange.close();
        });
        server.start();
        try {
            var base = URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/v1");
            var client = new EmbeddingClient(base, null, "bge-m3");
            assertArrayEquals(new double[]{1, 0}, client.embed(List.of("支付回调")).getFirst());
        } finally {
            server.stop(0);
        }
    }

    @Test
    void sortsProviderResponseByIndexAndSendsBatch() throws Exception {
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/embeddings",
                e -> {
                    var request = Json.MAPPER.readTree(e.getRequestBody());
                    String body =
                            request.path("model").asText().equals("qwen3-embedding-4b")
                                    && request.path("input").size() == 2
                                    && "Bearer test-key".equals(e.getRequestHeaders().getFirst("Authorization"))
                                    ? "{\"data\":[{\"index\":1,\"embedding\":[0,1]},{\"index\":0,\"embedding\":[1,0]}]}"
                                    : "{}";
                    byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
                    e.sendResponseHeaders(200, bytes.length);
                    e.getResponseBody().write(bytes);
                    e.close();
                });
        server.start();
        try {
            var client =
                    new EmbeddingClient(
                            URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/"),
                            "test-key",
                            "qwen3-embedding-4b");
            var v = client.embed(List.of("文档A", "文档B"));
            assertArrayEquals(new double[]{1, 0}, v.get(0));
            assertArrayEquals(new double[]{0, 1}, v.get(1));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void rejectsMissingVectorsAndDoesNotExposeProviderError() throws Exception {
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/embeddings",
                e -> {
                    byte[] b = "test-secret".getBytes();
                    e.sendResponseHeaders(401, b.length);
                    e.getResponseBody().write(b);
                    e.close();
                });
        server.start();
        try {
            var client =
                    new EmbeddingClient(
                            URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/"),
                            "test-secret",
                            "model");
            var error =
                    assertThrows(EmbeddingClient.UpstreamException.class, () -> client.embed(List.of("a")));
            assertFalse(error.getMessage().contains("test-secret"));
            assertTrue(error.getMessage().contains("401"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void malformedEmbeddingPayloadsAreRejected() throws Exception {
        for (String body :
                List.of(
                        "{\"data\":[]}",
                        "{\"data\":[{\"index\":0,\"embedding\":[0,0]}]}",
                        "{\"data\":[{\"index\":4,\"embedding\":[1,0]}]}",
                        "{\"data\":[{\"index\":4294967296,\"embedding\":[1,0]}]}")) {
            var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.createContext(
                    "/embeddings",
                    e -> {
                        byte[] b = body.getBytes();
                        e.sendResponseHeaders(200, b.length);
                        e.getResponseBody().write(b);
                        e.close();
                    });
            server.start();
            try {
                var client =
                        new EmbeddingClient(
                                URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/"),
                                "key",
                                "model");
                assertThrows(EmbeddingClient.UpstreamException.class, () -> client.embed(List.of("a")));
            } finally {
                server.stop(0);
            }
        }
    }
}
