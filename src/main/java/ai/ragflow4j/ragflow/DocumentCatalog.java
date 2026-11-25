package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.file.*;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

/**
 * 独立演示部署的文档目录，承担原版 DocumentDAO 的存活校验和外部 ID→全局文档 ID 映射。
 */
public final class DocumentCatalog {
    private final Path path;
    private final Path datasetPath;
    private final ObjectMapper json = new ObjectMapper();
    private Map<String, Map<String, Object>> documents;
    private Map<String, Map<String, Object>> datasets;

    public DocumentCatalog(Path path) throws Exception {
        this.path = path;
        datasetPath = path.resolveSibling("datasets.json");
        Files.createDirectories(path.toAbsolutePath().getParent());
        documents = read(path);
        datasets = read(datasetPath);
    }

    private Map<String, Map<String, Object>> read(Path source) throws Exception {
        return Files.exists(source) ? json.readValue(source.toFile(), new TypeReference<>() {})
                : new LinkedHashMap<>();
    }

    /**
     * 将数据集与外部文档 ID 组合成确定性内部 ID；相同文档重新入库仍得到同一标识。
     */
    public static String identity(String dataset, String externalId) {
        return UUID.nameUUIDFromBytes(
                        (dataset + "\0" + externalId).getBytes(java.nio.charset.StandardCharsets.UTF_8))
                .toString()
                .replace("-", "");
    }

    public static String sourceDigest(byte[] input) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(input));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(impossible);
        }
    }

    public synchronized void put(String dataset, String externalId, Map<String, Object> details)
            throws Exception {
        Map<String, Object> entry = new LinkedHashMap<>(details);
        entry.put("datasetId", dataset);
        entry.put("docId", externalId);
        entry.put("internalDocId", identity(dataset, externalId));
        Map<String, Map<String, Object>> next = new LinkedHashMap<>(documents);
        next.put(identity(dataset, externalId), entry);
        save(path, next);
        documents = next;
    }

    /** 同一个数据集的新文档一次落盘；调用者持有业务写锁。 */
    public synchronized void putNewBatch(String dataset, Map<String, Map<String, Object>> entries)
            throws Exception {
        Map<String, Map<String, Object>> next = new LinkedHashMap<>(documents);
        for (var item : entries.entrySet()) {
            String id = identity(dataset, item.getKey());
            if (next.containsKey(id)) throw new IllegalArgumentException("batch only accepts new documents");
            Map<String, Object> entry = new LinkedHashMap<>(item.getValue());
            entry.put("datasetId", dataset);
            entry.put("docId", item.getKey());
            entry.put("internalDocId", id);
            next.put(id, entry);
        }
        save(path, next);
        documents = next;
    }

    public synchronized void delete(String dataset, String externalId) throws Exception {
        Map<String, Map<String, Object>> next = new LinkedHashMap<>(documents);
        next.remove(identity(dataset, externalId));
        save(path, next);
        documents = next;
    }

    /**
     * 临时文件写成功后原子替换目录文件，最后才更新内存，避免失败时暴露未落盘状态。
     */
    private void save(Path destination, Map<String, Map<String, Object>> next) throws Exception {
        Path temp = Files.createTempFile(destination.toAbsolutePath().getParent(), "catalog-", ".tmp");
        try {
            json.writeValue(temp.toFile(), next);
            Files.move(temp, destination, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    /** Existing API-created corpora stay visible without an entry in datasets.json. */
    synchronized List<Map<String, Object>> datasets() {
        Map<String, Long> counts = new LinkedHashMap<>();
        for (var document : documents.values())
            counts.merge(document.get("datasetId").toString(), 1L, Long::sum);
        Set<String> ids = new LinkedHashSet<>(datasets.keySet());
        ids.addAll(counts.keySet());
        List<Map<String, Object>> result = new ArrayList<>();
        for (String id : ids) {
            var entry = new LinkedHashMap<>(datasets.getOrDefault(id, Map.of("name", id)));
            entry.put("datasetId", id);
            entry.put("documentCount", counts.getOrDefault(id, 0L));
            entry.put("registered", datasets.containsKey(id));
            result.add(entry);
        }
        return result;
    }

    synchronized Map<String, Object> dataset(String id) {
        return datasets.getOrDefault(id, Map.of());
    }

    synchronized void putDataset(String id, Map<String, Object> details) throws Exception {
        var next = new LinkedHashMap<>(datasets);
        next.put(id, Map.copyOf(details));
        save(datasetPath, next);
        datasets = next;
    }

    synchronized void deleteEmptyDataset(String id) throws Exception {
        if (!list(id).isEmpty()) throw new IllegalArgumentException("请先移除知识库里的文档");
        if (!datasets.containsKey(id)) return;
        var next = new LinkedHashMap<>(datasets);
        next.remove(id);
        save(datasetPath, next);
        datasets = next;
    }

    public synchronized List<Map<String, Object>> list(String dataset) {
        return documents.values().stream()
                .filter(d -> dataset.equals(d.get("datasetId")))
                .map(d -> (Map<String, Object>) new LinkedHashMap<>(d))
                .toList();
    }

    public synchronized Set<String> existingIds(String dataset) {
        Set<String> ids = new HashSet<>();
        for (var d : list(dataset)) ids.add(d.get("internalDocId").toString());
        return ids;
    }

    public synchronized Map<String, Object> get(String dataset, String docId) {
        return documents.get(identity(dataset, docId));
    }
}
