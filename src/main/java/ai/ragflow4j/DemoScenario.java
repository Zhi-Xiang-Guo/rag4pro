package ai.ragflow4j;

import ai.ragflow4j.ragflow.CorpusService;
import ai.ragflow4j.ragflow.DocumentCatalog;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** A versioned, synthetic runbook shipped with the application. */
final class DemoScenario {
    record Source(String docId, String title, String version, String section, String text) {
        Map<String, Object> metadata() {
            return Map.of("version", version, "section", section);
        }

        Map<String, Object> indexRequest() {
            return Map.of("docId", docId, "title", title, "text", text, "fileType", "txt",
                    "language", "Chinese", "chunkTokenSize", 0, "delimiters", List.of("\n"),
                    "metadata", metadata());
        }

        boolean matches(Map<String, Object> indexed) {
            String sourceHash = DocumentCatalog.sourceDigest(text.getBytes(StandardCharsets.UTF_8));
            return title.equals(indexed.get("title")) && metadata().equals(indexed.get("metadata"))
                    && sourceHash.equals(indexed.get("sourceSha256"));
        }
    }

    record Example(String label, String question, String version) {}

    record Fixture(int schemaVersion, List<Source> documents, List<Example> examples) {}

    static final class Conflict extends IllegalStateException {
        Conflict() {
            this("示例数据集存在同名文档，请检查目录后重试。");
        }

        Conflict(String message) {
            super(message);
        }
    }

    private final Fixture fixture;
    private final String revision;
    private final String datasetId;

    DemoScenario() throws IOException {
        try (var resource = Objects.requireNonNull(getClass().getResourceAsStream("/demo/payment.json"))) {
            byte[] payload = resource.readAllBytes();
            fixture = Json.MAPPER.readValue(payload, Fixture.class);
            if (fixture.schemaVersion() != 2) throw new IOException("Unsupported demo fixture revision");
            revision = DocumentCatalog.sourceDigest(payload);
            datasetId = "payment-handbook-" + revision.substring(0, 12);
        }
    }

    Map<String, Object> describe(CorpusService corpus) {
        return describe(corpus.documents(datasetId));
    }

    private Map<String, Object> describe(List<Map<String, Object>> indexed) {
        return Map.of("datasetId", datasetId, "revision", revision, "examples", fixture.examples(),
                "documentCount", fixture.documents().size(), "indexedCount", fixture.documents().size() - missing(indexed).size());
    }

    Map<String, Object> load(CorpusService corpus) throws Exception {
        // Hold the same lock as catalog writes, so two first-time visitors cannot index the same batch.
        synchronized (corpus) {
            var indexed = corpus.documents(datasetId);
            var pending = missing(indexed);
            if (!pending.isEmpty()) {
                corpus.indexBatch(Map.of("datasetId", datasetId,
                        "documents", pending.stream().map(Source::indexRequest).toList()));
            }
            return describe(corpus.documents(datasetId));
        }
    }

    Map<String, Object> answer(CorpusService corpus, Map<String, Object> input) throws Exception {
        String question = CorpusService.required(input, "question");
        String version = CorpusService.required(input, "version");
        if (fixture.documents().stream().noneMatch(doc -> doc.version().equals(version))) {
            throw new IllegalArgumentException("请选择手册版本 v1 或 v2");
        }
        if (question.codePointCount(0, question.length()) > 500) {
            throw new IllegalArgumentException("问题最多 500 字");
        }
        if (!missing(corpus.documents(datasetId)).isEmpty()) {
            throw new Conflict("请先建立示例索引。");
        }
        // The user selects a runbook version. Both retrieval branches inherit the same scope.
        return corpus.answer(Map.of("datasetId", datasetId, "question", question, "answerMode", "extractive",
                "strategy", "rrf", "topK", 8, "candidateK", 64, "evaluationTrace", true,
                "metadataFilter", Map.of("method", "manual",
                        "manual", List.of(Map.of("key", "version", "op", "=", "value", version)))));
    }

    Source source(String docId) {
        return fixture.documents().stream().filter(doc -> doc.docId().equals(docId)).findFirst().orElse(null);
    }

    private List<Source> missing(List<Map<String, Object>> indexed) {
        for (var row : indexed) {
            var original = source(Objects.toString(row.get("docId"), ""));
            if (original == null || !original.matches(row)) {
                throw new Conflict();
            }
        }
        return fixture.documents().stream()
                .filter(doc -> indexed.stream().noneMatch(row -> doc.docId().equals(row.get("docId"))))
                .toList();
    }
}
