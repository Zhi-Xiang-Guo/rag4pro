package ai.ragflow4j;

import ai.ragflow4j.ragflow.*;

import java.net.*;
import java.nio.file.Path;
import java.util.List;

import static ai.ragflow4j.Settings.get;

/**
 * 应用装配入口：创建模型客户端、分词器和存储，再交给业务服务与 HTTP 入口。
 *
 * <p>阅读时先看这里有哪些依赖，再跟随 StandardServer → CorpusService 进入具体业务。
 */
public final class Main {
    public static void main(String[] args) throws Exception {
        if (args.length > 0 && args[0].equals("--check-runtime")) {
            try (var analyzer = new NativeTokenizer()) {
                if (analyzer.tokenize("退款政策", "Chinese").isEmpty())
                    throw new IllegalStateException("Native analyzer returned empty result");
                if (new Cl100k().count("hello world") != 2 || new QwenBpe().count("hello") == 0)
                    throw new IllegalStateException("BPE resources unavailable");
                System.out.println("Packaged runtime resources: PASS");
            }
            return;
        }
        String embeddingModel = get("MODEL", "bge-m3");
        int embeddingMaxTokens = Integer.parseInt(get("EMBEDDING_MAX_TOKENS",
                EmbeddingClient.isQwenEmbedding(embeddingModel) ? "32768" : "8192"));
        var embeddings =
                new EmbeddingClient(
                        URI.create(get("BASE_URL", "http://127.0.0.1:11434/v1")),
                        get("API_KEY", ""),
                        embeddingModel, embeddingMaxTokens);
        // 建索引与查询共用 embedding 模型；索引适配器额外提供批量处理和模型窗口裁剪。
        var indexEmbedder =
                new IndexingPipeline.Embedder() {
                    @Override
                    public List<double[]> embed(List<String> texts) throws Exception {
                        return embeddings.embedForIngestion(texts);
                    }

                    @Override
                    public int maxTokens() {
                        return embeddingMaxTokens;
                    }

                    @Override
                    public String trim(String text) {
                        return embeddings.trim(text);
                    }
                };
        var nativeTokenizer = new NativeTokenizer();
        var store =
                new ElasticsearchStore(
                        URI.create(get("ES_URL", "http://127.0.0.1:19200")),
                        get("ES_INDEX", "evidence_rag"));
        store.initialize();
        Path dataPath = Path.of(get("DATA_DIR", "data"));
        var catalog = new DocumentCatalog(dataPath.resolve("documents.json"));
        int contextLength = Integer.parseInt(get("CHAT_CONTEXT", "8192"));
        URI ollama = URI.create(get("CHAT_URL", "http://127.0.0.1:11434"));
        var chatModel =
                new LocalChatModel(
                        ollama,
                        get("CHAT_MODEL", "qwen3:4b"),
                        contextLength);
        var extractiveModel = new LocalChatModel(ollama, get("EXTRACTIVE_MODEL", chatModel.model()),
                Math.min(contextLength, 8192));
        // 手动注入依赖：CorpusService 只需要拿到已构造的对象，不负责读取环境变量。
        var corpus =
                new CorpusService(
                        store,
                        catalog,
                        nativeTokenizer,
                        embeddings,
                        indexEmbedder,
                        chatModel,
                        contextLength,
                        new ContentVectorCache(dataPath.resolve("chunk-cache")),
                        extractiveModel);
        String rerankUrl = get("RERANK_URL", "");
        if (!rerankUrl.isBlank()) {
            String rerankModel = get("RERANK_MODEL", "");
            if (rerankModel.isBlank())
                throw new IllegalArgumentException(
                        "EVIDENCE_RAG_RERANK_MODEL is required with EVIDENCE_RAG_RERANK_URL");
            corpus.configureReranker(
                    new ConfiguredReranker(
                            URI.create(rerankUrl), get("RERANK_API_KEY", ""), rerankModel));
        }
        var server =
                new StandardServer(
                        new InetSocketAddress(
                                get("HOST", "127.0.0.1"),
                                Integer.parseInt(get("PORT", "18080"))),
                        corpus);
        Runtime.getRuntime()
                .addShutdownHook(
                        new Thread(
                                () -> {
                                    server.close();
                                    nativeTokenizer.close();
                                }));
        server.start();
        System.out.println(
                "Evidence RAG for Java ready. Embedding="
                        + embeddingModel
                        + ", chat="
                        + chatModel.model());
    }
}
