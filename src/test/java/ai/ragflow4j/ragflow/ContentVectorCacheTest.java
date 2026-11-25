package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.concurrent.Executors;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ContentVectorCacheTest {
    @TempDir
    Path directory;

    private static final class MutableClock extends Clock {
        Instant now = Instant.parse("2026-01-01T00:00:00Z");

        public ZoneId getZone() {
            return ZoneOffset.UTC;
        }

        public Clock withZone(ZoneId zone) {
            return this;
        }

        public Instant instant() {
            return now;
        }
    }

    @Test
    void keysMatchOriginalGoXxhashAndUtf8LengthPrefixes() {
        assertEquals("kc:emb:462598fc5ac601cf", ContentVectorCache.key("model", "chunk", "text"));
        assertEquals("kc:emb:7dd52abf18dfab12", ContentVectorCache.key("模型", "块", "正文🙂"));
        assertEquals("kc:emb:26f3500d1723f84e", ContentVectorCache.key("ab", "c", ""));
        assertEquals("kc:emb:ccfdb525c9c57e06", ContentVectorCache.key("a", "bc", ""));
        assertEquals("", ContentVectorCache.key("", "chunk", "text"));
        assertEquals("", ContentVectorCache.key("model", "", "text"));
    }

    @Test
    void survivesRestartAndHitsDoNotExtendSevenDayTtl() {
        MutableClock clock = new MutableClock();
        var first = new ContentVectorCache(directory, clock);
        String key = ContentVectorCache.key("model", "chunk", "text");
        first.put(key, new double[]{1, 2}, "");
        var restarted = new ContentVectorCache(directory, clock);
        clock.now = clock.now.plus(Duration.ofDays(6));
        double[] hit = restarted.get(key);
        assertArrayEquals(new double[]{1, 2}, hit);
        hit[0] = 999;
        assertArrayEquals(new double[]{1, 2}, restarted.get(key));
        clock.now = clock.now.plus(Duration.ofDays(1));
        assertNull(restarted.get(key));
        assertFalse(Files.exists(directory.resolve(key + ".json")));
    }

    @Test
    void corruptEntryAndUnavailableDirectoryAreMisses() throws Exception {
        var cache = new ContentVectorCache(directory);
        String key = ContentVectorCache.key("model", "chunk", "text");
        cache.put(key, new double[]{1, 2}, "");
        Files.writeString(directory.resolve(key + ".json"), "broken-json");
        assertNull(cache.get(key));
        Path blocked = directory.resolve("a-file");
        Files.writeString(blocked, "not a directory");
        var unavailable = new ContentVectorCache(blocked);
        assertDoesNotThrow(() -> unavailable.put(key, new double[]{1, 2}, "task"));
        assertNull(unavailable.get(key));
        assertThrows(java.io.IOException.class, () -> unavailable.purgeTask("task"));
    }

    @Test
    void taskPurgeReclaimsOnlyRegisteredMissesAndCanRetryPartialDeletion() throws Exception {
        var cache = new ContentVectorCache(directory);
        String first = ContentVectorCache.key("model", "one", "text");
        String second = ContentVectorCache.key("model", "two", "text");
        String unscoped = ContentVectorCache.key("model", "three", "text");
        cache.put(first, new double[]{1}, "task");
        cache.put(second, new double[]{2}, "task");
        cache.put(unscoped, new double[]{3}, "");
        Path obstruction = directory.resolve(second + ".json");
        Files.delete(obstruction);
        Files.createDirectory(obstruction);
        Files.writeString(obstruction.resolve("child"), "obstruction");
        assertThrows(java.io.IOException.class, () -> cache.purgeTask("task"));
        try (var files = Files.list(directory)) {
            assertEquals(
                    1, files.filter(path -> path.getFileName().toString().startsWith("manifest-")).count());
        }
        Files.delete(obstruction.resolve("child"));
        Files.delete(obstruction);
        cache.purgeTask("task");
        assertNull(cache.get(first));
        assertNull(cache.get(second));
        assertArrayEquals(new double[]{3}, cache.get(unscoped));
        assertDoesNotThrow(() -> cache.purgeTask("task"));
    }

    @Test
    void concurrentInstancesDoNotLoseTaskManifestMembers() throws Exception {
        try (var workers = Executors.newFixedThreadPool(4)) {
            var futures = new java.util.ArrayList<java.util.concurrent.Future<?>>();
            for (int i = 0; i < 30; i++) {
                int index = i;
                futures.add(
                        workers.submit(
                                () ->
                                        new ContentVectorCache(directory)
                                                .put(
                                                        ContentVectorCache.key("model", "chunk" + index, "text"),
                                                        new double[]{index},
                                                        "task")));
            }
            for (var future : futures) future.get();
        }
        var cache = new ContentVectorCache(directory);
        for (int i = 0; i < 30; i++)
            assertArrayEquals(
                    new double[]{i}, cache.get(ContentVectorCache.key("model", "chunk" + i, "text")));
        cache.purgeTask("task");
        try (var files = Files.list(directory)) {
            assertEquals(
                    0, files.filter(path -> path.getFileName().toString().endsWith(".json")).count());
        }
    }
}
