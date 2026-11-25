// Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
// http://www.apache.org/licenses/LICENSE-2.0
package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.file.*;
import java.util.*;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

/**
 * Go query_builder_test.go 与 term_weight_test.go 的固定向量，使用真实 C++ 分词器。
 */
class QueryBuilderTest {
    private static NativeTokenizer tokenizer;
    private static QueryBuilder query;
    private static TermWeightDealer dealer;

    @BeforeAll
    static void open() {
        tokenizer = new NativeTokenizer();
        dealer = new TermWeightDealer(tokenizer);
        query = new QueryBuilder(tokenizer, dealer, new Synonym());
    }

    @AfterAll
    static void close() {
        if (tokenizer != null) tokenizer.close();
    }

    @Test
    void originalChineseHeuristic() {
        for (String text :
                List.of("", "中文字符", "hello world", "hello world test", "Hello, world!", "这 是 一个 测试"))
            assertTrue(query.isChinese(text), text);
        assertFalse(query.isChinese("this is a long english sentence"));
        assertFalse(query.isChinese("hello world 中文 测试 多个"));
    }

    @Test
    void originalEscapesAndStopwordVectors() {
        assertEquals("test\\: colon", query.subSpecialChar("test: colon"));
        assertEquals("\\{braces\\}", query.subSpecialChar("{braces}"));
        assertEquals("path\\/to\\/file", query.subSpecialChar("path/to/file"));
        assertEquals("\\[brackets\\]", query.subSpecialChar("[brackets]"));
        assertEquals("a\\-b\\-c", query.subSpecialChar("a-b-c"));
        assertEquals("a\\?b", query.subSpecialChar("a?b"));
        assertEquals("its", query.subSpecialChar("it's"));
        assertEquals("x", query.subSpecialChar(" ' x ' "));
        assertEquals("操作", query.rmWWW("请问如何操作"));
        assertEquals("安装", query.rmWWW("怎么办安装"));
        assertEquals(" this", query.rmWWW("what is this"));
        assertEquals(" you", query.rmWWW("who are you"));
        assertEquals(" the problem", query.rmWWW("请问what is the problem"));
        assertEquals("请问", query.rmWWW("请问"));
        assertEquals(" cat on mat", query.rmWWW("the cat is on a mat"));
    }

    @Test
    void normalizationUsesOriginalCharacterTable() {
        assertEquals("hello 世界 test 测试", query.addSpaceBetweenEngZh("hello世界test测试"));
        assertEquals("abc123 测试", query.addSpaceBetweenEngZh("abc123测试"));
        assertEquals("测试 abc123", query.addSpaceBetweenEngZh("测试abc123"));
        assertEquals("Hello World!123", query.strFullWidth2HalfWidth("Ｈｅｌｌｏ　Ｗｏｒｌｄ！123"));
        assertEquals("繁体中文测试", query.traditional2Simplified("繁體中文測試"));
        assertEquals("我学习中文已经三年了", query.traditional2Simplified("我學習中文已經三年了"));
        assertFalse(query.needFineGrainedTokenize("ab"));
        assertFalse(query.needFineGrainedTokenize("hello-world"));
        assertTrue(query.needFineGrainedTokenize("中华人民共和国"));
    }

    @Test
    void originalTokenMergeAndSplitVectors() {
        assertEquals(List.of(), dealer.tokenMerge(List.of()));
        assertEquals(List.of("a b c"), dealer.tokenMerge(List.of("a", "b", "c")));
        assertEquals(List.of("a", "hello", "b"), dealer.tokenMerge(List.of("a", "hello", "b")));
        assertEquals(List.of("多 工位"), dealer.tokenMerge(List.of("多", "工位")));
        assertEquals(
                List.of("a b", "c d", "e f"), dealer.tokenMerge(List.of("a", "b", "c", "d", "e", "f")));
        assertEquals(List.of("machine learning algorithm"), dealer.split("machine learning algorithm"));
        assertEquals(List.of("hello", "世界", "world"), dealer.split("hello 世界 world"));
        assertEquals(List.of(""), dealer.split(""));
        assertEquals(List.of("hello world"), dealer.split("hello    world"));
    }

    @Test
    void exactSharedOovFixture() throws Exception {
        try (var input = getClass().getResourceAsStream("/ragflow/term_weight_oov.json")) {
            for (var vector : new ObjectMapper().readTree(input)) {
                Double actual = TermWeightDealer.alphabeticOOVFrequency(vector.get("term").asText());
                if (vector.get("frequency").isNull()) assertNull(actual);
                else assertEquals(vector.get("frequency").asDouble(), actual);
            }
        }
    }

    @Test
    void originalDictionaryLoadingAndWeights(@TempDir Path directory) throws Exception {
        Files.writeString(
                directory.resolve("ner.json"), "{\"北京\":\"loca\",\"腾讯\":\"corp\",\"func\":\"func\"}");
        Files.writeString(
                directory.resolve("term.freq"), "hello\t100\nworld\t200\ntest\t50\nzero\nbad\tx\n");
        TermWeightDealer local = new TermWeightDealer(tokenizer, directory);
        assertEquals("loca", local.ner("北京"));
        assertEquals(100, local.getDF().get("hello"));
        assertEquals(0, local.getDF().get("zero"));
        assertFalse(local.getDF().containsKey("bad"));
        assertEquals(List.of("hello", "func", "world"), local.split("hello func world"));
        List<Reranker.TermWeight> weights =
                local.weights(List.of("alpha", "alpha", "beta", "北京"), false, "");
        assertEquals(1, weights.stream().mapToDouble(Reranker.TermWeight::weight).sum(), 1e-14);
        assertEquals(weights.get(0).weight(), weights.get(1).weight());
        assertTrue(local.weights(List.of("请问", "您", "你"), true, "").isEmpty());
    }

    @Test
    void questionBranchesAndOriginalMetadata() {
        for (String text :
                List.of("请问如何安装软件", "How to install software", "hello世界", "alpha beta gamma delta")) {
            QueryBuilder.QuestionResult result = query.question(text, "test", 0.5, "");
            assertNotNull(result.expression(), text);
            assertFalse(result.expression().matchingText().isEmpty(), text);
            assertFalse(result.keywords().isEmpty(), text);
            assertEquals(text, result.expression().extraOptions().get("original_query"));
            assertEquals(100, result.expression().topN());
        }
        assertNull(query.question("", "test", 0.5, "").expression());
        assertEquals(
                0.5,
                query
                        .question("请问如何安装软件", "test", 0.5, "")
                        .expression()
                        .extraOptions()
                        .get("minimum_should_match"));
        assertFalse(
                query
                        .question("alpha beta gamma delta", "test", 0.5, "")
                        .expression()
                        .extraOptions()
                        .containsKey("minimum_should_match"));
        assertEquals(
                "\"hello\" \"world\"",
                query.paragraph("ignored", List.of("hello", "world", "test"), 2).matchingText());
    }

    @Test
    void rawWeightFormattingRetainsSmallBoosts() {
        assertEquals("1", QueryBuilder.goFloat(1));
        assertEquals("0.0005", QueryBuilder.goFloat(0.0005));
        assertEquals("5e-05", QueryBuilder.goFloat(0.00005));
        assertEquals("0.034123456789", QueryBuilder.goFloat(0.034123456789));
        assertEquals("0.0312", QueryBuilder.boost(0.03125));
        assertEquals("0.0938", QueryBuilder.boost(0.09375));
    }

    @Test
    void originalSlovakLanguageParityVector() {
        String original = "daňové priznanie pre živnostníkov";
        QueryBuilder.QuestionResult folded = query.question(original, "test", 0.5, "Slovak");
        assertTrue(
                folded.expression().matchingText().codePoints().allMatch(codePoint -> codePoint < 128));
        assertTrue(folded.expression().matchingText().contains("danove"));
        assertEquals(original, folded.expression().extraOptions().get("original_query"));
        assertTrue(
                query
                        .question(original, "test", 0.5, "")
                        .expression()
                        .matchingText()
                        .codePoints()
                        .anyMatch(codePoint -> codePoint > 127));
    }
}
