// Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
// http://www.apache.org/licenses/LICENSE-2.0
package ai.ragflow4j.ragflow;

import static ai.ragflow4j.ragflow.Reranker.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

import org.junit.jupiter.api.Test;

/**
 * 原 Go reranker_*_test.go 的已知向量和分支。词权重依赖固定为1，独立验证融合算法。
 */
class RerankerTest {
    private static final TokenSource TOKENS =
            new TokenSource() {
                public List<String> question(String query, String language) {
                    return split(query);
                }

                public List<TermWeight> weights(List<String> tokens, String language) {
                    return tokens.stream().map(token -> new TermWeight(token, 1)).toList();
                }
            };
    private static final String CONTENT = "content_ltks";

    @Test
    void cosineKnownVectorsAndDegenerateInputs() {
        assertEquals(1, cosineSimilarity(new double[]{1, 2, 3}, new double[]{1, 2, 3}), 1e-9);
        assertEquals(0, cosineSimilarity(new double[]{1, 0}, new double[]{0, 1}));
        assertEquals(-1, cosineSimilarity(new double[]{1, 0}, new double[]{-1, 0}));
        assertEquals(1, cosineSimilarity(new double[]{1, 2}, new double[]{2, 4}), 1e-9);
        assertEquals(-1, cosineSimilarity(new double[]{1, 2}, new double[]{-2, -4}), 1e-9);
        assertEquals(0, cosineSimilarity(new double[]{1}, new double[]{1, 2}));
        assertEquals(0, cosineSimilarity(new double[0], new double[0]));
        assertEquals(0, cosineSimilarity(new double[]{0, 0}, new double[]{1, 2}));
        assertTrue(
                Double.isNaN(
                        cosineSimilarity(new double[]{Double.POSITIVE_INFINITY, 0}, new double[]{1, 0})));
    }

    @Test
    void tokenDictionaryOrderRepetitionAndAdjacency() {
        Map<String, Double> weights =
                tokensToDict(List.of("alpha", "beta", "alpha", "gamma"), TOKENS, "");
        assertEquals(
                List.of("alpha", "alphabeta", "beta", "betaalpha", "alphagamma", "gamma"),
                new ArrayList<>(weights.keySet()));
        assertEquals(0.8, weights.get("alpha"));
        assertEquals(0.6, weights.get("alphabeta"));
        assertEquals(0.4, weights.get("beta"));
        assertTrue(tokensToDict(List.of("alpha"), null, "").isEmpty());
    }

    @Test
    void queryL1OverlapUsesPresenceAndSmoothing() {
        assertEquals(0, tokenDictSimilarity(Map.of(), Map.of("a", 1.0), null));
        assertEquals(
                1,
                tokenDictSimilarity(
                        Map.of("a", 1.0, "b", 1.0), Map.of("a", 5.0, "b", 5.0), List.of("a", "b")));
        assertEquals(
                (1e-9 + 1) / (1e-9 + 4),
                tokenDictSimilarity(Map.of("a", 1.0, "b", 3.0), Map.of("a", 9.0), List.of("a", "b")));
        assertEquals(1e-9 / (1e-9 + 1), tokenDictSimilarity(Map.of("a", 1.0), Map.of("z", 1.0), null));
        assertArrayEquals(
                new double[]{0, 0},
                tokenSimilarity(List.of("alpha"), List.of(List.of("alpha"), List.of("beta")), null, ""));
        assertArrayEquals(
                new double[]{1},
                tokenSimilarity(List.of("alpha", "beta"), List.of(List.of("alpha", "beta")), TOKENS, ""));
    }

    @Test
    void tokenAssemblyPreservesOriginalFieldRepetitions() {
        Map<String, Object> chunk =
                Map.of(
                        CONTENT,
                        "a b a",
                        "title_tks",
                        "t1 t2",
                        "important_kwd",
                        List.of("i1", "i2"),
                        "question_tks",
                        "q1 q2");
        assertEquals(
                List.of(
                        "a", "b", "t1", "t2", "t1", "t2", "i1", "i2", "i1", "i2", "i1", "i2", "i1", "i2", "i1",
                        "i2", "q1", "q2", "q1", "q2", "q1", "q2", "q1", "q2", "q1", "q2", "q1", "q2"),
                buildRerankTokens(chunk, CONTENT, 2, 5, 6));
        assertEquals(
                List.of("a", "b", "t1", "t2", "i1", "i2", "q1", "q2"),
                buildRerankTokens(chunk, CONTENT, 1, 1, 1));
        assertEquals(
                List.of("x", "y", "x", "x", "x", "x", "x", "x"),
                buildRerankTokens(
                        Map.of(
                                CONTENT,
                                "x x y",
                                "title_tks",
                                "x x",
                                "important_kwd",
                                List.of("x", "x"),
                                "question_tks",
                                "x x"),
                        CONTENT,
                        1,
                        1,
                        1));
        assertEquals(
                List.of("a b", "c"),
                buildRerankTokens(Map.of("important_kwd", List.of("a b", 3, "c")), CONTENT, 1, 1, 1));
    }

    @Test
    void standardVectorAndPagerankKnownValues() {
        List<Map<String, Object>> chunks =
                List.of(
                        Map.of(CONTENT, "alpha", "q_2_vec", new double[]{1, 0}, "pagerank_fea", 0.5),
                        Map.of(CONTENT, "beta", "q_2_vec", List.of(0.0, 1.0)));
        Result scores =
                rerankStandard(
                        chunks, null, new double[]{1, 0}, "alpha", 0.3, 0.7, CONTENT, null, null, "");
        assertArrayEquals(new double[]{1, 0}, scores.vsim());
        assertArrayEquals(new double[]{1.2, 0}, scores.sim(), 1e-9);
        assertArrayEquals(new double[]{0, 0}, scores.tsim());
        assertEquals(0.3, DEFAULT_VECTOR_WEIGHT);
        assertEquals(0.7, DEFAULT_TOKEN_WEIGHT);
        assertEquals(1, DEFAULT_VECTOR_ONLY_WEIGHT);
    }

    @Test
    void allZeroStandardVectorsUseUnweightedTokensWithoutAliasing() {
        Result result =
                rerankStandard(
                        List.of(Map.of(CONTENT, "alpha", "pagerank_fea", 2.5)),
                        List.of("alpha"),
                        null,
                        "alpha",
                        0.3,
                        0.7,
                        CONTENT,
                        TOKENS,
                        null,
                        "");
        assertArrayEquals(new double[]{1}, result.tsim());
        assertArrayEquals(new double[]{3.5}, result.sim());
        assertNotSame(result.sim(), result.tsim());
        assertArrayEquals(new double[]{0}, result.vsim());
    }

    @Test
    void hybridLengthMismatchRetainsFailureBoundary() {
        assertThrows(
                ArrayIndexOutOfBoundsException.class,
                () ->
                        hybridSimilarity(
                                new double[]{1, 0},
                                List.of(new double[]{1, 0}),
                                List.of("alpha"),
                                List.of(List.of("alpha"), List.of("beta")),
                                0.3,
                                0.7,
                                null,
                                ""));
    }

    @Test
    void knnMappingFallbackAndMissingRemainAligned() {
        List<String> ids = List.of("a", "b", "c");
        Result result =
                rerankWithKNN(
                        List.of(Map.of(CONTENT, "unused"), Map.of(CONTENT, "beta", "pagerank_fea", 9)),
                        ids,
                        Map.of("a", Map.of(CONTENT, "alpha")),
                        Map.of("a", 0.9, "c", 0.4),
                        "alpha",
                        0.3,
                        0.7,
                        CONTENT,
                        TOKENS,
                        null,
                        "");
        assertArrayEquals(new double[]{0.9, 0, 0.4}, result.vsim());
        assertEquals(0.3 + 0.7 * 0.9, result.sim()[0]);
        assertEquals(0.3 * result.tsim()[1], result.sim()[1]);
        assertEquals(0, result.tsim()[2]);
        assertEquals(0.7 * 0.4, result.sim()[2]);
        Result zero =
                rerankWithKNN(
                        null,
                        List.of("a"),
                        Map.of("a", Map.of(CONTENT, "alpha")),
                        null,
                        "alpha",
                        0.3,
                        0.7,
                        CONTENT,
                        TOKENS,
                        null,
                        "");
        assertEquals(0.3, zero.sim()[0]);
    }

    @Test
    void knnNormalizesKeywordInPlaceAndIncludesQuestions() {
        Map<String, Object> chunk =
                new HashMap<>(Map.of(CONTENT, "alpha", "important_kwd", "delta", "question_tks", "quux"));
        Result result =
                rerankWithKNN(
                        null, List.of("a"), Map.of("a", chunk), null, "quux", 1, 0, CONTENT, TOKENS, null, "");
        assertArrayEquals(new String[]{"delta"}, (String[]) chunk.get("important_kwd"));
        assertEquals(1, result.tsim()[0]);
    }

    @Test
    void normalizationOriginalKnownVectorsInPlace() {
        double[][] inputs = {
                {10, -3, 0},
                {100, 50, 75},
                {-1, -5, -3},
                {0.9, 0.1, 0.5},
                {0.8, 0.8, 0.8},
                {1},
                {0, 1, 0.42},
                {5},
                {-3},
                {5, 5, 5},
                {-2, -2, -2},
                {-2, -1.9999}
        };
        double[][] expected = {
                {1, 0, 3.0 / 13},
                {1, 0, 0.5},
                {1, 0, 0.5},
                {0.9, 0.1, 0.5},
                {0.8, 0.8, 0.8},
                {1},
                {0, 1, 0.42},
                {1},
                {0},
                {1, 1, 1},
                {0, 0, 0},
                {0, 0}
        };
        for (int i = 0; i < inputs.length; i++) {
            assertSame(inputs[i], normalizeRerankScores(inputs[i]));
            assertArrayEquals(expected[i], inputs[i], 1e-9);
        }
        assertNull(normalizeRerankScores(null));
        assertTrue(Double.isNaN(normalizeRerankScores(new double[]{Double.NaN, 2})[0]));
    }

    @Test
    void rankFeaturesOriginalVectorsAndFormats() {
        Map<String, Double> query = Map.of("tag_a", 2.0, "tag_b", 1.0, "pagerank_fea", 7.0);
        for (Object tags :
                List.of("{\"tag_a\":0.6,\"tag_b\":0.8}", Map.of("tag_a", 0.6, "tag_b", 0.8))) {
            Map<String, Object> chunk = Map.of("tag_feas", tags, "pagerank_fea", 0.25);
            double expected = 0.3 + (2 / Math.sqrt(5)) * 10 + 0.25;
            assertEquals(
                    expected, applyRankFeatureScores(List.of(chunk), new double[]{0.3}, query)[0], 1e-9);
            assertEquals(
                    expected,
                    applyRankFeatureScoresForIDs(
                            List.of("c1"), Map.of("c1", chunk), new double[]{0.3}, query)[0],
                    1e-9);
        }
        for (Object tags : List.of("{\"tag_a\":0.6,", "{}", Map.of("tag_a", 0.0), List.of(1, 2))) {
            assertEquals(
                    0.55,
                    applyRankFeatureScores(
                            List.of(Map.of("tag_feas", tags, "pagerank_fea", 0.25)), new double[]{0.3}, query)[
                            0]);
        }
        assertEquals(
                0.55,
                applyRankFeatureScores(
                        List.of(Map.of("tag_feas", Map.of("tag_a", 1.0), "pagerank_fea", 0.25)),
                        new double[]{0.3},
                        Map.of("pagerank_fea", 7.0))[0]);
        assertArrayEquals(
                new double[]{8, 4},
                applyRankFeatureScores(
                        List.of(Map.of("pagerank_fea", "7"), Map.of("pagerank_fea", 3)),
                        new double[]{1, 1},
                        null));
        assertArrayEquals(
                new double[]{1, 5},
                applyRankFeatureScores(List.of(Map.of("pagerank_fea", 1)), new double[]{0, 5}, null));
        assertArrayEquals(
                new double[]{2, 7},
                applyRankFeatureScoresForIDs(
                        List.of("a"), Map.of("a", Map.of("pagerank_fea", 2)), new double[]{0, 7}, null));
        assertArrayEquals(
                new double[]{0.2},
                applyRankFeatureScoresForIDs(List.of("missing"), Map.of(), new double[]{0.2}, query));
    }

    @Test
    void unknownTagsEnterDenominator() {
        assertEquals(
                10 / Math.sqrt(2),
                applyRankFeatureScores(
                        List.of(Map.of("tag_feas", Map.of("tag_a", 1, "unknown", 1))),
                        new double[]{0},
                        Map.of("tag_a", 1.0))[0]);
    }

    @Test
    void modelUsesNaturalTextAndReordersResponseIndices() throws Exception {
        String natural = "<p>La sécurité des données : voir l'annexe A.</p>";
        List<Map<String, Object>> chunks =
                List.of(
                        Map.of(CONTENT, "alpha", "content_with_weight", natural, "pagerank_fea", 0.5),
                        Map.of(CONTENT, "beta", "content_with_weight", natural));
        Result result =
                rerankByModel(
                        (query, docs) -> {
                            assertEquals(List.of(natural, natural), docs);
                            return List.of(
                                    new ModelScore(1, -3),
                                    new ModelScore(0, 10),
                                    new ModelScore(-1, 99),
                                    new ModelScore(5, 99));
                        },
                        chunks,
                        List.of("a", "b"),
                        Map.of("a", chunks.get(0), "b", chunks.get(1)),
                        "alpha",
                        0.3,
                        0.7,
                        CONTENT,
                        TOKENS,
                        null,
                        "");
        assertArrayEquals(new double[]{1, 0}, result.vsim());
        assertEquals(1.5, result.sim()[0]);
        assertTrue(result.tsim()[1] < 1e-6);
    }

    @Test
    void modelFallbackFormattingEmptyIdsAndErrorPropagation() throws Exception {
        Map<String, Object> chunk =
                Map.of(
                        CONTENT,
                        "alpha beta",
                        "title_tks",
                        "gamma",
                        "important_kwd",
                        List.of("delta"),
                        "question_tks",
                        "epsilon zeta");
        rerankByModel(
                (query, docs) -> {
                    assertEquals(List.of("alpha beta gamma delta epsilon zeta"), docs);
                    return List.of();
                },
                List.of(chunk),
                List.of("missing"),
                null,
                "alpha",
                0.3,
                0.7,
                CONTENT,
                TOKENS,
                null,
                "");
        assertEquals("(alpha) beta!", removeRedundantSpaces("( alpha ) beta !"));
        assertEquals("привет мир", removeRedundantSpaces("привет мир"));
        Result empty =
                rerankByModel(
                        (query, docs) -> {
                            assertTrue(docs.isEmpty());
                            return List.of();
                        },
                        List.of(chunk),
                        null,
                        null,
                        "alpha",
                        0.3,
                        0.7,
                        CONTENT,
                        TOKENS,
                        null,
                        "");
        assertEquals(0, empty.sim().length);
        Exception failure = new Exception("model failed");
        assertSame(
                failure,
                assertThrows(
                        Exception.class,
                        () ->
                                rerankByModel(
                                        (query, docs) -> {
                                            throw failure;
                                        },
                                        List.of(chunk),
                                        List.of("a"),
                                        null,
                                        "alpha",
                                        0.3,
                                        0.7,
                                        CONTENT,
                                        TOKENS,
                                        null,
                                        "")));
    }
}
