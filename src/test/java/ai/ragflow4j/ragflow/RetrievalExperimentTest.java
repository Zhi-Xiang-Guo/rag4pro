package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.atomic.*;
import org.junit.jupiter.api.Test;

class RetrievalExperimentTest {
    @Test void rrfHandlesDisjointRecallAndDeduplicatesWithinBranch() {
        var lists = List.of(List.of("a", "a", "b"), List.of("b", "c"));
        var scores = ReciprocalRankFusion.scores(lists, 60);
        assertEquals(1.0/62 + 1.0/61, scores.get("b"), 1e-12);
        assertEquals(1.0/61, scores.get("a"), 1e-12);
        assertEquals(List.of("b", "a", "c"), ReciprocalRankFusion.rank(lists,60));
        assertThrows(IllegalArgumentException.class, () -> ReciprocalRankFusion.rank(lists,0));
    }
    @Test void cacheExpiresEvictsAndDoesNotExposeMutableVectorsOrCacheFailures() throws Exception {
        AtomicLong now = new AtomicLong(); AtomicInteger calls = new AtomicInteger();
        var cache = new QueryEmbeddingCache(q -> {
            calls.incrementAndGet();
            if (q.equals("failure")) throw new IllegalStateException();
            return new double[]{1,2};
        }, 1, Duration.ofNanos(10), now::get);
        cache.query("a")[0]=99;
        assertEquals(1,cache.query("a")[0]); assertEquals(1,calls.get());
        now.set(11); cache.query("a"); assertEquals(2,calls.get());
        cache.query("b"); cache.query("a"); assertEquals(4,calls.get());
        assertThrows(IllegalStateException.class, () -> cache.query("failure"));
        assertThrows(IllegalStateException.class, () -> cache.query("failure"));
        assertEquals(6,calls.get());
    }
    @Test void batchIndexingMatchesSingleDocumentVectorsIncludingTitleBlendAndTrim() throws Exception {
        IndexingPipeline.LexicalTokenizer lexical = new IndexingPipeline.LexicalTokenizer() {
            public String tokenize(String s,String language){return s.toLowerCase();}
            public String fineGrainedTokenize(String s,String language){return s;}
        };
        List<Integer> batches = new ArrayList<>();
        IndexingPipeline.Embedder encoder = new IndexingPipeline.Embedder() {
            public List<double[]> embed(List<String> texts) {
                batches.add(texts.size());
                return texts.stream().map(s -> new double[]{s.length(),s.hashCode()%1000}).toList();
            }
            public String trim(String s) { return s.substring(0,Math.min(8,s.length())); }
        };
        var pipeline=new IndexingPipeline(lexical,new Cl100k(),encoder);
        var config=IndexingPipeline.Config.defaults("English");
        var a=List.<Map<String,Object>>of(Map.of("text","long duplicated body", "id","a"));
        var b=List.<Map<String,Object>>of(Map.of("text","long duplicated body", "id","b"));
        var expectedA=pipeline.process("A.txt",a,config); var expectedB=pipeline.process("B.txt",b,config);
        batches.clear();
        var actual=pipeline.processBatch(List.of(new IndexingPipeline.Document("A.txt",a,config),new IndexingPipeline.Document("B.txt",b,config)));
        assertEquals(List.of(3),batches); // two titles and one unique trimmed body
        for (int i=0;i<2;i++) {
            var expected=(i==0?expectedA:expectedB).getFirst(); var got=actual.get(i).getFirst();
            assertArrayEquals((double[])expected.get("q_2_vec"),(double[])got.get("q_2_vec"));
            assertEquals(expected.get("content_ltks"),got.get("content_ltks"));
            assertEquals(expected.get("id"),got.get("id"));
        }
    }
}
