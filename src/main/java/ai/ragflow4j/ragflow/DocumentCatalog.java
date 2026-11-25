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
    private final ObjectMapper json = new ObjectMapper();
    private Map<String, Map<String, Object>> documents;

    public DocumentCatalog(Path path) throws Exception {
        this.path = path;
        Files.createDirectories(path.toAbsolutePath().getParent());
        documents =
                Files.exists(path)
                        ? json.readValue(path.toFile(), new TypeReference<>() {
                })
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
        save(next);
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
        save(next);
    }

    public synchronized void delete(String dataset, String externalId) throws Exception {
        Map<String, Map<String, Object>> next = new LinkedHashMap<>(documents);
        next.remove(identity(dataset, externalId));
        save(next);
    }

    /**
     * 临时文件写成功后原子替换目录文件，最后才更新内存，避免失败时暴露未落盘状态。
     */
    private void save(Map<String, Map<String, Object>> next) throws Exception {
        Path temp = Files.createTempFile(path.toAbsolutePath().getParent(), "catalog-", ".tmp");
        try {
            json.writeValue(temp.toFile(), next);
            Files.move(temp, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            documents = next;
        } finally {
            Files.deleteIfExists(temp);
        }
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
