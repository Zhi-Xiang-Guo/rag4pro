package ai.ragflow4j;

import ai.ragflow4j.ragflow.CorpusService;
import ai.ragflow4j.ragflow.DocumentParser;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Local single-instance queue with durable status and a document revision per upload. */
final class DocumentIngestion implements AutoCloseable {
    enum Status { QUEUED, PARSING, INDEXING, READY, FAILED }

    record Job(String id, String datasetId, String docId, String title, String filename, String format,
               Map<String, Object> options, String expectedVersion, Status status, int attempts,
               String updatedAt, String error) {
        Job state(Status next, String message) {
            return new Job(id, datasetId, docId, title, filename, format, options, expectedVersion,
                    next, next == Status.PARSING ? attempts + 1 : attempts, Instant.now().toString(),
                    message);
        }

        boolean pending() {
            return status == Status.QUEUED || status == Status.PARSING || status == Status.INDEXING;
        }
    }

    static final class Conflict extends IllegalStateException {
        Conflict(String message) { super(message); }
    }

    private static final int MAX_PENDING = 32;
    private final CorpusService corpus;
    private final Path root;
    private final Map<String, Job> jobs = new LinkedHashMap<>();
    private final ExecutorService worker = Executors.newSingleThreadExecutor(
            Thread.ofVirtual().name("document-ingestion").factory());

    DocumentIngestion(CorpusService corpus, Path root) throws IOException {
        this.corpus = corpus;
        this.root = root;
        Files.createDirectories(root);
        try (var dirs = Files.list(root)) {
            for (Path directory : dirs.sorted().toList()) {
                Path manifest = directory.resolve("job.json");
                if (!Files.isRegularFile(manifest)) continue;
                Job job = Json.MAPPER.readValue(manifest.toFile(), Job.class);
                jobs.put(job.id(), job);
            }
        }
    }

    synchronized void start() throws IOException {
        for (Job job : List.copyOf(jobs.values())) {
            if (job.pending()) enqueue(job.state(Status.QUEUED, ""));
        }
    }

    synchronized Job submit(Map<String, String> query, byte[] file) throws IOException {
        String dataset = require(query, "datasetId");
        String filename = require(query, "filename");
        String format = DocumentParser.format(filename);
        if (filename.length() > 255 || filename.chars().anyMatch(Character::isISOControl))
            throw new IllegalArgumentException("文件名超出允许范围");
        if (file.length == 0) throw new IllegalArgumentException("文件为空");
        if (file.length > DocumentParser.MAX_BYTES) throw new IllegalArgumentException("文件超过 16 MiB");
        if (!corpus.hasDataset(dataset)) throw new IllegalArgumentException("请先创建知识库");
        String docId = query.getOrDefault("docId", "");
        if (docId.isBlank()) docId = UUID.randomUUID().toString();
        if (docId.length() > 256 || docId.chars().anyMatch(Character::isISOControl))
            throw new IllegalArgumentException("文档 ID 超出允许范围");
        requireCapacity(dataset, docId);
        var document = corpus.document(dataset, docId);
        String version = CorpusService.documentVersion(document);
        if (query.containsKey("expectedVersion") && !version.equals(query.get("expectedVersion")))
            throw new Conflict("文档已变更，请打开当前版本后重新提交");
        String title = query.getOrDefault("title", filename).strip();
        if (title.isEmpty() || title.length() > 255) throw new IllegalArgumentException("文档标题须为 1–255 字符");
        Map<String, Object> options = new LinkedHashMap<>();
        for (String key : List.of("chunkStrategy", "chunkTokenSize", "overlapPercent")) {
            if (document != null && document.containsKey(key)) options.put(key, document.get(key));
            if (!query.getOrDefault(key, "").isBlank()) options.put(key, parseOption(key, query.get(key)));
        }
        String id = UUID.randomUUID().toString();
        var job = new Job(id, dataset, docId, title, filename, format, options,
                version, Status.QUEUED, 0, Instant.now().toString(), "");
        Files.createDirectory(root.resolve(id));
        Files.write(original(job), file);
        enqueue(job);
        return job;
    }

    synchronized Job retry(String id) throws IOException {
        Job job = get(id);
        if (job == null) return null;
        if (job.status() != Status.FAILED) throw new Conflict("只可重试失败的入库任务");
        if (!corpus.hasDataset(job.datasetId())) throw new Conflict("知识库已不存在");
        String active = CorpusService.documentVersion(corpus.document(job.datasetId(), job.docId()));
        if (!active.equals(job.expectedVersion()) && !active.equals(job.id()))
            throw new Conflict("文档已变更，请基于当前版本重新提交");
        requireCapacity(job.datasetId(), job.docId());
        Job queued = job.state(Status.QUEUED, "");
        enqueue(queued);
        return queued;
    }

    synchronized Job get(String id) {
        return jobs.get(id);
    }

    synchronized List<Job> list(String dataset) {
        return jobs.values().stream().filter(job -> dataset.equals(job.datasetId()))
                .sorted(Comparator.comparing(Job::updatedAt).reversed()).toList();
    }

    synchronized boolean pending(String dataset, String docId) {
        return jobs.values().stream().anyMatch(job -> job.pending() && dataset.equals(job.datasetId())
                && (docId == null || docId.equals(job.docId())));
    }

    synchronized void deleteDocument(String dataset, String docId) throws Exception {
        if (pending(dataset, docId)) throw new Conflict("文档正处理入库任务");
        corpus.delete(dataset, docId);
    }

    synchronized void deleteEmptyDataset(String dataset) throws Exception {
        if (pending(dataset, null)) throw new Conflict("知识库有索引任务正在进行");
        corpus.deleteEmptyDataset(dataset);
    }

    private void requireCapacity(String dataset, String docId) {
        if (pending(dataset, docId)) throw new Conflict("该文档有正在处理的入库任务");
        if (jobs.values().stream().filter(Job::pending).count() >= MAX_PENDING)
            throw new Conflict("入库队列已满，请稍后再试");
    }

    private void enqueue(Job job) throws IOException {
        persist(job);
        worker.submit(() -> ingest(job.id()));
    }

    private synchronized void change(String id, Status status, String error) throws IOException {
        persist(jobs.get(id).state(status, error));
    }

    private void persist(Job job) throws IOException {
        Path directory = root.resolve(job.id());
        Path temp = Files.createTempFile(directory, "job-", ".tmp");
        try {
            Json.MAPPER.writeValue(temp.toFile(), job);
            Files.move(temp, directory.resolve("job.json"), StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
            jobs.put(job.id(), job);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    private void ingest(String id) {
        Job job = get(id);
        try {
            change(id, Status.PARSING, "");
            var source = corpus.sources().find(job.datasetId(), job.docId(), id);
            if (source == null) {
                byte[] bytes = Files.readAllBytes(original(job));
                source = corpus.sources().save(job.datasetId(), job.docId(), id, job.filename(),
                        job.format(), bytes, DocumentParser.parse(job.format(), bytes));
            }
            change(id, Status.INDEXING, "");
            Map<String, Object> request = new LinkedHashMap<>(job.options());
            request.put("datasetId", job.datasetId());
            request.put("docId", job.docId());
            request.put("title", job.title());
            corpus.indexSource(request, source, job.expectedVersion());
            change(id, Status.READY, "");
        } catch (Exception failure) {
            if (worker.isShutdown()) return;
            try {
                String message = Objects.toString(failure.getMessage(), "入库失败");
                change(id, Status.FAILED, message.substring(0, Math.min(500, message.length())));
            } catch (IOException persistenceFailure) {
                System.err.println("Could not persist ingestion failure for " + id + ": "
                        + persistenceFailure.getClass().getSimpleName());
            }
            return;
        }
        try {
            Files.deleteIfExists(original(job));
        } catch (IOException cleanupFailure) {
            System.err.println("Could not remove staged upload for " + id);
        }
    }

    private Path original(Job job) {
        return root.resolve(job.id()).resolve("upload");
    }

    private static Object parseOption(String key, String value) {
        return switch (key) {
            case "chunkTokenSize" -> Integer.valueOf(value);
            case "overlapPercent" -> Double.valueOf(value);
            default -> value;
        };
    }

    private static String require(Map<String, String> query, String key) {
        String value = query.getOrDefault(key, "").strip();
        if (value.isEmpty()) throw new IllegalArgumentException(key + " 不能为空");
        return value;
    }

    @Override public void close() {
        worker.shutdownNow();
    }
}
