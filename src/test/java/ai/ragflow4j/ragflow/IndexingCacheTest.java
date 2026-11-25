package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class IndexingCacheTest {
    @TempDir
    Path directory;
    private static final ContextBuilder.TokenCounter COUNTER =
            new ContextBuilder.TokenCounter() {
                public int count(String text) {
                    return text.length();
                }

                public String trim(String text, int limit) {
                    return text.substring(0, Math.min(limit, text.length()));
                }
            };
    private static final IndexingPipeline.LexicalTokenizer LEXICAL =
            new IndexingPipeline.LexicalTokenizer() {
                public String tokenize(String text, String language) {
                    return text;
                }

                public String fineGrainedTokenize(String text, String language) {
                    return text;
                }
            };

    private static final class Encoder implements IndexingPipeline.Embedder {
        final List<List<String>> calls = new ArrayList<>();
        int trimLimit = 1000;
        int batchSize = 16;
        String failOn = "";

        public int batchSize() {
            return batchSize;
        }

        public String trim(String text) {
            return COUNTER.trim(text, trimLimit);
        }

        public List<double[]> embed(List<String> texts) {
            calls.add(List.copyOf(texts));
            if (!failOn.isEmpty() && texts.contains(failOn))
                throw new IllegalStateException("model failed");
            return texts.stream()
                    .map(text -> text.startsWith("Title") ? new double[]{10, 20} : new double[]{1, 2})
                    .toList();
        }
    }

    private IndexingPipeline pipeline(Encoder encoder, ContentVectorCache cache, String model) {
        return new IndexingPipeline(LEXICAL, COUNTER, encoder, cache, model);
    }

    private static IndexingPipeline.Config config(double weight, List<String> fields) {
        return new IndexingPipeline.Config("English", true, true, weight, fields);
    }

    private static double[] vector(List<Map<String, Object>> chunks) {
        return (double[]) chunks.getFirst().get("q_2_vec");
    }

    @Test
    void cachedContentIsUnweightedAndTitleIsRecomputedEvenOnAllHits() throws Exception {
        var cache = new ContentVectorCache(directory);
        var encoder = new Encoder();
        var pipeline = pipeline(encoder, cache, "model");
        var chunks = List.<Map<String, Object>>of(Map.of("id", "chunk", "text", "body"));
        assertArrayEquals(
                new double[]{1.9, 3.8},
                vector(pipeline.process("Title.txt", chunks, config(.1, List.of("text")))),
                1e-12);
        assertArrayEquals(
                new double[]{1, 2}, cache.get(ContentVectorCache.key("model", "chunk", "body")));
        assertArrayEquals(
                new double[]{5.5, 11},
                vector(pipeline.process("Title.md", chunks, config(.5, List.of("text")))));
        assertEquals(
                List.of(List.of("Title.txt"), List.of("body"), List.of("Title.md")), encoder.calls);
    }

    @Test
    void keyUsesActualCleanedTrimmedSelectedTextAndModelIdentity() throws Exception {
        var cache = new ContentVectorCache(directory);
        var encoder = new Encoder();
        encoder.trimLimit = 3;
        var pipeline = pipeline(encoder, cache, "model-a");
        var chunks =
                List.<Map<String, Object>>of(
                        Map.of("id", "chunk", "text", "abcdef", "summary", "abcXYZ", "questions", "xyz"));
        pipeline.process("", chunks, config(0, List.of("text")));
        pipeline.process("", chunks, config(.5, List.of("summary")));
        assertEquals(1, encoder.calls.size());
        pipeline.process("", chunks, config(0, List.of("questions")));
        assertEquals(List.of("xyz"), encoder.calls.getLast());
        pipeline(encoder, cache, "model-b").process("", chunks, config(0, List.of("text")));
        assertEquals(3, encoder.calls.size());
        encoder.trimLimit = 4;
        pipeline.process("", chunks, config(0, List.of("text")));
        assertEquals(List.of("abcd"), encoder.calls.getLast());
    }

    @Test
    void missingEmptyOrNonStringChunkIdentityAndEmptyModelSkipCaching() throws Exception {
        var cache = new ContentVectorCache(directory);
        for (Map<String, Object> chunk :
                List.<Map<String, Object>>of(
                        Map.of("text", "body"),
                        Map.of("id", "", "text", "body"),
                        Map.of("id", 123, "text", "body"))) {
            var encoder = new Encoder();
            var pipeline = pipeline(encoder, cache, "model");
            pipeline.process("", List.of(chunk), config(0, List.of("text")));
            pipeline.process("", List.of(chunk), config(0, List.of("text")));
            assertEquals(2, encoder.calls.size());
        }
        var encoder = new Encoder();
        var pipeline = pipeline(encoder, cache, "");
        var chunks = List.<Map<String, Object>>of(Map.of("id", "chunk", "text", "body"));
        pipeline.process("", chunks, config(0, List.of("text")));
        pipeline.process("", chunks, config(0, List.of("text")));
        assertEquals(2, encoder.calls.size());
    }

    @Test
    void failedLaterBatchDoesNotWriteEarlierBatchToCache() {
        var cache = new ContentVectorCache(directory);
        var encoder = new Encoder();
        encoder.batchSize = 1;
        encoder.failOn = "bad";
        var pipeline = pipeline(encoder, cache, "model");
        var chunks =
                List.<Map<String, Object>>of(
                        Map.of("id", "one", "text", "good"), Map.of("id", "two", "text", "bad"));
        assertThrows(
                IllegalStateException.class,
                () -> pipeline.process("Title", chunks, config(.1, List.of("text")), "task"));
        assertNull(cache.get(ContentVectorCache.key("model", "one", "good")));
        assertNull(cache.get(ContentVectorCache.key("model", "two", "bad")));
    }

    @Test
    void wrongDimensionCachedVectorFailsBlendAndEmptyTableSkipsTitle() throws Exception {
        var cache = new ContentVectorCache(directory);
        cache.put(ContentVectorCache.key("model", "chunk", "body"), new double[]{1}, "");
        var encoder = new Encoder();
        var pipeline = pipeline(encoder, cache, "model");
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        pipeline.process(
                                "Title",
                                List.of(Map.of("id", "chunk", "text", "body")),
                                config(.1, List.of("text"))));
        assertEquals(List.of(List.of("Title")), encoder.calls);
        encoder.calls.clear();
        pipeline.process(
                "Title",
                List.of(Map.of("id", "table", "text", "<table></table>")),
                config(.1, List.of("text")));
        assertEquals(List.of(), encoder.calls);
    }

    @Test
    void taskManifestIsPurgedAfterPersistAndCacheHitsDoNotCreateAnotherManifest() throws Exception {
        var cache = new ContentVectorCache(directory);
        var encoder = new Encoder();
        var pipeline = pipeline(encoder, cache, "model");
        var chunks = List.<Map<String, Object>>of(Map.of("id", "chunk", "text", "body"));
        pipeline.process("", chunks, config(0, List.of("text")), "task-a");
        pipeline.process("", chunks, config(0, List.of("text")), "task-b");
        cache.purgeTask("task-b");
        assertNotNull(cache.get(ContentVectorCache.key("model", "chunk", "body")));
        cache.purgeTask("task-a");
        assertNull(cache.get(ContentVectorCache.key("model", "chunk", "body")));
    }
}
