// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.time.Clock;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import net.jpountz.xxhash.XXHashFactory;

/**
 * 正文向量的磁盘缓存，对应 ingestion/chunkcache 的缓存条目与任务清单。
 *
 * <p>保存的是未混入标题的向量；任务失败后可在 TTL 内复用，持久化成功后再按任务清单清理。
 */
public final class ContentVectorCache {
    public static final Duration TTL = Duration.ofDays(7);
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final ConcurrentHashMap<Path, Object> ROOT_LOCKS = new ConcurrentHashMap<>();
    private final Path root;
    private final Clock clock;
    private final Object monitor;
    private long nextSweep;

    public ContentVectorCache(Path root) {
        this(root, Clock.systemUTC());
    }

    ContentVectorCache(Path root, Clock clock) {
        this.root = root.toAbsolutePath().normalize();
        this.clock = clock;
        this.monitor = ROOT_LOCKS.computeIfAbsent(this.root, ignored -> new Object());
    }

    /**
     * 与 chunkcache.Key 使用相同的 UTF-8 长度前缀、xxhash64 种子及十六进制表示。
     */
    public static String key(String modelId, String chunkId, String embeddedText) {
        if (modelId == null || modelId.isEmpty() || chunkId == null || chunkId.isEmpty()) return "";
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (String field : List.of("emb", modelId, chunkId, embeddedText)) {
            byte[] encoded = field.getBytes(StandardCharsets.UTF_8);
            bytes.writeBytes((encoded.length + ":").getBytes(StandardCharsets.US_ASCII));
            bytes.writeBytes(encoded);
        }
        byte[] input = bytes.toByteArray();
        long hash = XXHashFactory.fastestJavaInstance().hash64().hash(input, 0, input.length, 0);
        return "kc:emb:" + Long.toUnsignedString(hash, 16);
    }

    /**
     * 读取异常或向量数据损坏视为缓存未命中，匹配 chunkcache.Get 的降级行为。
     */
    public double[] get(String key) {
        if (!validKey(key)) return null;
        try {
            return locked(
                    () -> {
                        JsonNode entry = readLive(root.resolve(key + ".json"));
                        if (entry == null) return null;
                        JsonNode value = entry.path("value");
                        if (!value.isArray() || value.isEmpty()) return null;
                        double[] vector = new double[value.size()];
                        for (int i = 0; i < vector.length; i++) {
                            JsonNode element = value.get(i);
                            // encoding/json permits null array entries and decodes them to float64 zero.
                            if (!element.isNull() && !element.isNumber()) return null;
                            vector[i] = element.isNull() ? 0 : element.doubleValue();
                            if (!Double.isFinite(vector[i])) return null;
                        }
                        return vector;
                    });
        } catch (IOException | RuntimeException unavailable) {
            return null;
        }
    }

    /**
     * 仅写入未加权的正文向量；缓存故障不阻断建索引。
     */
    public void put(String key, double[] vector, String taskId) {
        if (!validKey(key) || vector == null || vector.length == 0) return;
        for (double value : vector) if (!Double.isFinite(value)) return;
        try {
            locked(
                    () -> {
                        long expires = clock.millis() + TTL.toMillis();
                        writeAtomic(
                                root.resolve(key + ".json"), Map.of("expiresAtMillis", expires, "value", vector));
                        if (taskId != null && !taskId.isEmpty()) {
                            Path path = manifestPath(taskId);
                            JsonNode previous = readLive(path);
                            LinkedHashSet<String> keys = new LinkedHashSet<>();
                            if (previous != null) {
                                for (JsonNode member : previous.path("keys")) {
                                    if (member.isTextual() && validKey(member.asText())) keys.add(member.asText());
                                }
                            }
                            keys.add(key);
                            writeAtomic(path, Map.of("expiresAtMillis", expires, "keys", keys));
                        }
                        return null;
                    });
        } catch (IOException | RuntimeException unavailable) {
            // The value and manifest each have a bounded TTL even when the second write fails.
        }
    }

    /**
     * 仅在文档持久化成功后调用；删除失败时保留任务清单，便于重试。
     */
    public void purgeTask(String taskId) throws IOException {
        if (taskId == null || taskId.isEmpty()) return;
        locked(
                () -> {
                    Path manifest = manifestPath(taskId);
                    JsonNode entry = readLive(manifest);
                    if (entry == null) return null;
                    JsonNode members = entry.path("keys");
                    if (!members.isArray()) throw new IOException("Malformed content cache task manifest");
                    List<String> keys = new ArrayList<>();
                    for (JsonNode member : members) {
                        if (!member.isTextual() || !validKey(member.asText()))
                            throw new IOException("Invalid content cache manifest member");
                        keys.add(member.asText());
                    }
                    for (String key : keys) Files.deleteIfExists(root.resolve(key + ".json"));
                    Files.deleteIfExists(manifest);
                    return null;
                });
    }

    private static boolean validKey(String key) {
        return key != null && key.matches("kc:emb:[0-9a-f]{1,16}");
    }

    private Path manifestPath(String taskId) {
        try {
            byte[] hash =
                    MessageDigest.getInstance("SHA-256").digest(taskId.getBytes(StandardCharsets.UTF_8));
            return root.resolve("manifest-" + HexFormat.of().formatHex(hash) + ".json");
        } catch (java.security.NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(impossible);
        }
    }

    private JsonNode readLive(Path path) throws IOException {
        if (!Files.exists(path)) return null;
        JsonNode entry = JSON.readTree(Files.readAllBytes(path));
        if (entry == null || !entry.path("expiresAtMillis").isIntegralNumber())
            throw new IOException("Malformed content cache entry");
        if (entry.path("expiresAtMillis").asLong() <= clock.millis()) {
            Files.deleteIfExists(path);
            return null;
        }
        return entry;
    }

    private void writeAtomic(Path destination, Object value) throws IOException {
        Path temporary = Files.createTempFile(root, ".cache-", ".tmp");
        try {
            Files.write(temporary, JSON.writeValueAsBytes(value));
            Files.move(
                    temporary,
                    destination,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    private interface IOOperation<T> {
        T run() throws IOException;
    }

    private <T> T locked(IOOperation<T> operation) throws IOException {
        synchronized (monitor) {
            Files.createDirectories(root);
            try (FileChannel channel =
                         FileChannel.open(
                                 root.resolve(".lock"), StandardOpenOption.CREATE, StandardOpenOption.WRITE);
                 var ignored = channel.lock()) {
                sweepExpired();
                return operation.run();
            }
        }
    }

    private void sweepExpired() throws IOException {
        long now = clock.millis();
        if (now < nextSweep) return;
        nextSweep = now + Duration.ofMinutes(1).toMillis();
        try (var paths = Files.list(root)) {
            for (Path path :
                    paths.filter(file -> file.getFileName().toString().endsWith(".json")).toList()) {
                try {
                    JsonNode entry = JSON.readTree(Files.readAllBytes(path));
                    if (entry != null
                            && entry.path("expiresAtMillis").isIntegralNumber()
                            && entry.path("expiresAtMillis").asLong() <= now) Files.deleteIfExists(path);
                } catch (IOException ignored) {
                    // Individual corrupt/unreadable entries remain misses without blocking other keys.
                }
            }
        }
    }
}
