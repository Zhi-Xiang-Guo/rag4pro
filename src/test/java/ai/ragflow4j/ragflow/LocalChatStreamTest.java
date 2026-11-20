package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class LocalChatStreamTest {
    @Test void openAiEventsAreDeliveredBeforeTheProviderCompletes() throws Exception {
        verify(true, "data: {\"choices\":[{\"delta\":{\"content\":\"first \"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"second\"},\"finish_reason\":\"stop\"}]}\n\n"
                        + "data: [DONE]\n\n");
    }

    @Test void ollamaLinesAreDeliveredBeforeTheProviderCompletes() throws Exception {
        verify(false, "{\"message\":{\"content\":\"first \"},\"done\":false}\n",
                "{\"message\":{\"content\":\"second\"},\"done\":true,\"eval_count\":2}\n");
    }

    private void verify(boolean openAi, String first, String last) throws Exception {
        var sentToClient = new CountDownLatch(1);
        var optedIntoStreaming = new java.util.concurrent.atomic.AtomicBoolean();
        var providerSawClientDelta = new java.util.concurrent.atomic.AtomicBoolean();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            optedIntoStreaming.set(new ObjectMapper().readTree(exchange.getRequestBody()).path("stream").asBoolean());
            exchange.sendResponseHeaders(200, 0);
            try (var output = exchange.getResponseBody()) {
                output.write(first.getBytes(StandardCharsets.UTF_8));
                output.flush();
                try {
                    providerSawClientDelta.set(sentToClient.await(5, TimeUnit.SECONDS));
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                }
                output.write(last.getBytes(StandardCharsets.UTF_8));
            }
        });
        server.start();
        try {
            URI endpoint = URI.create("http://127.0.0.1:" + server.getAddress().getPort());
            LocalChatModel model = openAi ? LocalChatModel.openAi(endpoint, "local-test", "fixture", 1024)
                    : new LocalChatModel(endpoint, "fixture", 1024);
            List<String> chunks = new ArrayList<>();
            var answer = model.stream(List.of(new ContextBuilder.Message("user", "question")), Map.of(),
                    new ChatPipeline.Stream() {
                        @Override public void delta(String text) {
                            chunks.add(text);
                            sentToClient.countDown();
                        }
                        @Override public void attach(InputStream upstream) { }
                    });
            assertTrue(optedIntoStreaming.get());
            assertTrue(providerSawClientDelta.get());
            assertEquals(List.of("first ", "second"), chunks);
            assertEquals("first second", answer.answer());
        } finally {
            server.stop(0);
        }
    }
}
