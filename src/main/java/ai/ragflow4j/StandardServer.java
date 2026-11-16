package ai.ragflow4j;

import ai.ragflow4j.ragflow.CorpusService;
import ai.ragflow4j.ragflow.DocumentSources;
import ai.ragflow4j.ragflow.LocalChatModel;
import com.fasterxml.jackson.core.type.TypeReference;
import com.sun.net.httpserver.*;

import java.io.IOException;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;

/**
 * 标准链路 HTTP 入口，返回真实的中间字段、查询 DSL、评分和引用。
 */
public final class StandardServer implements AutoCloseable {
    private final HttpServer server;
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final CorpusService corpus;
    private final DemoScenario demo;
    // 包括排队在内的上限；CPU/模型/目录仍有串行阶段，避免无界虚拟线程堆积。
    private final Semaphore admission;
    private final String apiToken;

    public StandardServer(InetSocketAddress address, CorpusService corpus) throws IOException {
        this(address, corpus, Integer.parseInt(Settings.get("MAX_INFLIGHT", "8")),
                Settings.get("HTTP_TOKEN", ""));
    }

    public StandardServer(InetSocketAddress address, CorpusService corpus, int maxInflight, String apiToken)
            throws IOException {
        if (maxInflight < 1 || maxInflight > 256) throw new IllegalArgumentException("maxInflight must be 1..256");
        this.corpus = corpus;
        this.demo = new DemoScenario();
        this.admission = new Semaphore(maxInflight);
        this.apiToken = Objects.requireNonNullElse(apiToken, "");
        server = HttpServer.create(address, 32);
        server.setExecutor(executor);
        server.createContext("/", this::handle);
    }

    public InetSocketAddress address() { return server.getAddress(); }

    public void start() {
        server.start();
    }

    private void handle(HttpExchange exchange) throws IOException {
        boolean admitted = false;
        String requestId = UUID.randomUUID().toString();
        exchange.getResponseHeaders().set("X-Request-Id", requestId);
        long started = System.nanoTime();
        try {
            String path = exchange.getRequestURI().getPath(), method = exchange.getRequestMethod();
            if (method.equals("GET") && servePage(exchange, path)) return;
            boolean health = path.equals("/api/health") && method.equals("GET");
            if (!health) {
                String authorization = Objects.toString(exchange.getRequestHeaders().getFirst("Authorization"), "");
                if (!apiToken.isBlank() && !java.security.MessageDigest.isEqual(
                        authorization.getBytes(StandardCharsets.UTF_8),
                        ("Bearer " + apiToken).getBytes(StandardCharsets.UTF_8))) {
                    respond(exchange, 401, Map.of("error", "unauthorized", "requestId", requestId));
                    return;
                }
                admitted = admission.tryAcquire();
                if (!admitted) {
                    exchange.getResponseHeaders().set("Retry-After", "1");
                    respond(exchange, 429, Map.of("error", "server busy", "requestId", requestId));
                    return;
                }
            }
            Map<String, String> query = query(exchange);
            String dataset = query.getOrDefault("datasetId", "demo");
            Object result;
            if (path.equals("/api/health") && method.equals("GET")) result = corpus.health();
            else if (path.equals("/api/demo") && method.equals("GET")) result = demo.describe(corpus);
            else if (path.equals("/api/demo") && method.equals("POST")) result = demo.load(corpus);
            else if (path.equals("/api/demo/source") && method.equals("GET")) {
                result = demo.source(query.get("docId"));
                if (result == null) {
                    respond(exchange, 404, Map.of("error", "示例文档不存在"));
                    return;
                }
            }
            else if ((path.equals("/api/document-source") || path.equals("/api/document-source/file"))
                    && method.equals("GET")) {
                String docId = CorpusService.required(new HashMap<>(query), "docId");
                var document = corpus.document(dataset, docId);
                String revision = query.getOrDefault("revision",
                        document == null ? "" : Objects.toString(document.get("sourceRevision"), ""));
                var version = document == null || revision.isBlank()
                        ? null : corpus.sources().find(dataset, docId, revision);
                if (version == null) {
                    respond(exchange, 404, Map.of("error", "这个版本没有保存原件"));
                    return;
                }
                if (path.endsWith("/file")) {
                    exchange.getResponseHeaders().set("Content-Disposition",
                            "attachment; filename*=UTF-8''" + URLEncoder.encode(version.name(), StandardCharsets.UTF_8)
                                    .replace("+", "%20"));
                    send(exchange, 200, DocumentSources.contentType(version.format()),
                            corpus.sources().original(dataset, docId, revision));
                    return;
                }
                result = version;
            }
            else if (path.equals("/api/datasets") && method.equals("GET"))
                result = corpus.datasets();
            else if (path.equals("/api/datasets") && method.equals("DELETE")) {
                corpus.deleteEmptyDataset(CorpusService.required(new HashMap<>(query), "datasetId"));
                result = Map.of("deleted", true);
            } else if (path.equals("/api/documents") && method.equals("GET"))
                result = corpus.documents(dataset);
            else if (path.equals("/api/chunks") && method.equals("GET"))
                result = corpus.chunks(dataset, query.get("docId"));
            else if (path.equals("/api/documents") && method.equals("DELETE")) {
                String doc = query.get("docId");
                if (doc == null || doc.isBlank()) throw new IllegalArgumentException("docId 不能为空");
                corpus.delete(dataset, doc);
                result = Map.of("deleted", true);
            } else if (method.equals("POST")
                    && Set.of("/api/datasets", "/api/documents", "/api/documents/batch", "/api/chunk-preview",
                            "/api/retrieve", "/api/retrieve/context", "/api/chat", "/api/demo/chat")
                    .contains(path)) {
                String contentType = exchange.getRequestHeaders().getFirst("Content-Type");
                if (contentType == null
                        || !contentType.split(";", 2)[0].strip().equalsIgnoreCase("application/json")) {
                    respond(exchange, 415, Map.of("error", "Content-Type 必须为 application/json"));
                    return;
                }
                byte[] bytes = exchange.getRequestBody().readNBytes(2 * 1024 * 1024 + 1);
                if (bytes.length > 2 * 1024 * 1024) {
                    respond(exchange, 413, Map.of("error", "请求体超过 2 MiB"));
                    return;
                }
                Map<String, Object> input = Json.MAPPER.readValue(bytes, new TypeReference<>() {
                });
                if (input == null) throw new IllegalArgumentException("JSON 请求体为空");
                // HTTP 层只处理协议与参数边界；三个业务入口都由 CorpusService 编排。
                // 初次阅读检索时，从 /api/retrieve 分支进入 corpus.retrieve 即可。
                result =
                        switch (path) {
                            case "/api/datasets" -> corpus.saveDataset(input);
                            case "/api/documents" -> corpus.index(input);
                            case "/api/documents/batch" -> corpus.indexBatch(input);
                            case "/api/chunk-preview" -> corpus.previewChunks(input);
                            case "/api/retrieve" -> corpus.retrieve(input);
                            case "/api/retrieve/context" -> corpus.retrieveContext(input);
                            case "/api/demo/chat" -> demo.answer(corpus, input);
                            default -> corpus.answer(input);
                        };
            } else {
                respond(exchange, 404, Map.of("error", "接口不存在或方法不支持"));
                return;
            }
            respond(exchange, 200, result);
        } catch (com.fasterxml.jackson.core.JsonProcessingException error) {
            respond(exchange, 400, Map.of("error", "JSON 格式无效"));
        } catch (IllegalArgumentException error) {
            respond(exchange, 400, Map.of("error", Objects.toString(error.getMessage(), "参数无效")));
        } catch (DemoScenario.Conflict error) {
            respond(exchange, 409, Map.of("error", error.getMessage()));
        } catch (LocalChatModel.Unavailable error) {
            respond(exchange, 503, Map.of("error", error.getMessage(), "code", "local_model_unavailable",
                    "attempts", error.attempts(), "requestId", requestId));
        } catch (LocalChatModel.InvalidSelection error) {
            respond(exchange, 502, Map.of("error", error.getMessage(), "code", "invalid_evidence_selection",
                    "attempts", error.attempts(), "requestId", requestId));
        } catch (Exception error) {
            System.err.println(
                    "Standard request failed: "
                            + error.getClass().getSimpleName()
                            + ": "
                            + safeMessage(error));
            respond(exchange, 502, Map.of("error", safeMessage(error)));
        } finally {
            if (admitted) admission.release();
            System.err.println("requestId=" + requestId + " method=" + exchange.getRequestMethod()
                    + " path=" + exchange.getRequestURI().getPath() + " durationMs="
                    + ((System.nanoTime() - started) / 1_000_000));
            exchange.close();
        }
    }

    private boolean servePage(HttpExchange exchange, String path) throws IOException {
        if (path.matches("/app-assets/assets/[A-Za-z0-9_.-]+")) {
            return staticResource(exchange, "/react-workspace/" + path.substring("/app-assets/".length()),
                    path.endsWith(".css") ? "text/css; charset=utf-8"
                            : "text/javascript; charset=utf-8");
        }
        boolean workspace = path.equals("/") || path.equals("/chat") || path.startsWith("/chat/")
                || path.equals("/knowledge") || path.startsWith("/knowledge/") || path.equals("/retrieve");
        if (workspace && staticResource(exchange, "/react-workspace/index.html", "text/html; charset=utf-8"))
            return true;
        String resource = workspace ? "/workspace/index.html" : switch (path) {
            case "/workspace-legacy" -> "/workspace/index.html";
            case "/workspace.css" -> "/workspace/style.css";
            case "/workspace.js" -> "/workspace/app.js";
            case "/demo" -> "/demo/index.html";
            case "/demo.css" -> "/demo/style.css";
            case "/demo.js" -> "/demo/app.js";
            case "/lab" -> "/index.html";
            default -> null;
        };
        if (resource == null) return false;
        String contentType = switch (path) {
            case "/demo.css", "/workspace.css" -> "text/css; charset=utf-8";
            case "/demo.js", "/workspace.js" -> "text/javascript; charset=utf-8";
            default -> "text/html; charset=utf-8";
        };
        return staticResource(exchange, resource, contentType);
    }

    private boolean staticResource(HttpExchange exchange, String resource, String contentType) throws IOException {
        try (var page = getClass().getResourceAsStream(resource)) {
            if (page == null) return false;
            send(exchange, 200, contentType, page.readAllBytes());
            return true;
        }
    }

    private String safeMessage(Exception error) {
        // 上游 HTTP 客户端只报告状态码，不把 Authorization/请求正文带入异常。
        String message = Objects.toString(error.getMessage(), "链路执行失败");
        String key = Settings.get("API_KEY", "");
        if (!key.isBlank()) message = message.replace(key, "[redacted]");
        return message.length() > 1500 ? message.substring(0, 1500) : message;
    }

    private Map<String, String> query(HttpExchange exchange) {
        Map<String, String> result = new HashMap<>();
        String raw = exchange.getRequestURI().getRawQuery();
        if (raw == null) return result;
        for (String piece : raw.split("&")) {
            String[] pair = piece.split("=", 2);
            result.put(
                    URLDecoder.decode(pair[0], StandardCharsets.UTF_8),
                    pair.length > 1 ? URLDecoder.decode(pair[1], StandardCharsets.UTF_8) : "");
        }
        return result;
    }

    private void respond(HttpExchange exchange, int status, Object result) throws IOException {
        send(
                exchange, status, "application/json; charset=utf-8", Json.MAPPER.writeValueAsBytes(result));
    }

    private void send(HttpExchange exchange, int status, String contentType, byte[] bytes)
            throws IOException {
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.getResponseHeaders().set("Cache-Control", "no-store");
        exchange.getResponseHeaders().set("X-Content-Type-Options", "nosniff");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
    }

    @Override
    public void close() {
        server.stop(1);
        executor.shutdownNow();
    }
}
