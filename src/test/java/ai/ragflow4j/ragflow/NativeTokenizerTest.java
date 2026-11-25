package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import java.util.concurrent.Executors;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

class NativeTokenizerTest {
    @Test
    void matchesOriginalCAbiOutputsAndDictionaryValues() {
        try (var tokenizer = new NativeTokenizer()) {
            assertEquals(
                    "the child were run and mous ate chees",
                    tokenizer.tokenize("The children were running and mice ate cheeses."));
            assertEquals("繁体中文 与 机器学习", tokenizer.tokenize("繁體中文與機器學習"));
            assertEquals("繁体 中文 与 机器 学习", tokenizer.fineGrainedTokenize("繁体中文 与 机器学习"));
            assertEquals("中华人民共和国 人工智能", tokenizer.tokenize("中华人民共和国人工智能"));
            assertEquals("中华 人民共和国 人工 智能", tokenizer.fineGrainedTokenize("中华人民共和国 人工智能"));
            assertEquals("abc123 hello 世界", tokenizer.tokenize("ＡＢＣ１２３ Hello 世界"));
            assertEquals(367879, tokenizer.freq("中国"));
            assertEquals("ns", tokenizer.tag("中国"));
            assertEquals(335, tokenizer.frequency("人工智能"));
            assertEquals("n", tokenizer.tag("人工智能"));
            assertEquals(0, tokenizer.freq("not_a_word_937393"));
            assertEquals("", tokenizer.tag("not_a_word_937393"));
            assertEquals("", tokenizer.tokenize(""));
            assertThrows(IllegalArgumentException.class, () -> tokenizer.tokenize("abc\0def"));
        }
    }

    @Test
    void serializesNativeStateAndResetsLanguageAcrossConcurrentCalls() throws Exception {
        try (var tokenizer = new NativeTokenizer();
             var pool = Executors.newFixedThreadPool(4)) {
            var calls =
                    IntStream.range(0, 40)
                            .mapToObj(
                                    i ->
                                            pool.submit(
                                                    () -> {
                                                        tokenizer.tokenize("žlutý kůň", "Czech");
                                                        return tokenizer.tokenize("children running", "English");
                                                    }))
                            .toList();
            for (var result : calls) assertEquals("child run", result.get());
        }
    }

    @Test
    void closesExactlyOnceAndRejectsUseAfterClose() {
        var tokenizer = new NativeTokenizer();
        tokenizer.close();
        tokenizer.close();
        assertThrows(IllegalStateException.class, () -> tokenizer.tokenize("word"));
        assertThrows(IllegalStateException.class, () -> tokenizer.frequency("word"));
        assertThrows(IllegalStateException.class, () -> tokenizer.tag("word"));
    }
}
