// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
// http://www.apache.org/licenses/LICENSE-2.0
package ai.ragflow4j.ragflow;

import static ai.ragflow4j.ragflow.MetadataFilter.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

import org.junit.jupiter.api.Test;

/**
 * metadata_filter_test.go/common.metadata_utils_test.go 的固定向量与边界。
 */
class MetadataFilterTest {
    private static final Map<String, Map<String, List<String>>> AUTHORS =
            Map.of("author", Map.of("Zhang San", List.of("doc-1"), "Li Si", List.of("doc-3")));

    private static List<String> apply(
            Map<String, Map<String, List<String>>> metadata, String key, Object value, String operator) {
        return applyMetaFilter(metadata, List.of(new Condition(key, value, operator)), "and").stream()
                .sorted()
                .toList();
    }

    private static Map<String, Object> manual(String key, Object value, String op) {
        Map<String, Object> condition = new LinkedHashMap<>();
        condition.put("key", key);
        condition.put("value", value);
        condition.put("op", op);
        return Map.of("method", "manual", "manual", List.of(condition));
    }

    @Test
    void originalEqualityAliasesAndStringPredicates() {
        for (String operator : List.of("=", "==", "is"))
            assertEquals(List.of("doc-1"), apply(AUTHORS, "author", "zhang san", operator));
        for (String operator : List.of("≠", "!=", "not is"))
            assertEquals(List.of("doc-3"), apply(AUTHORS, "author", "zhang san", operator));
        var metadata =
                Map.of("title", Map.of("hello world", List.of("doc1"), "goodbye", List.of("doc2")));
        assertEquals(List.of("doc1"), apply(metadata, "title", "HELLO", "contains"));
        assertEquals(List.of("doc2"), apply(metadata, "title", "hello", "not contains"));
        assertEquals(List.of("doc1"), apply(metadata, "title", "HELLO", "start with"));
        assertEquals(List.of("doc1"), apply(metadata, "title", "WORLD", "end with"));
        assertEquals(List.of(), apply(metadata, "title", "hello world", "unsupported"));
    }

    @Test
    void originalSetArraysCommaAndInvalidValues() {
        var metadata =
                Map.of(
                        "category",
                        Map.of(
                                "A",
                                List.of("doc1"),
                                "B",
                                List.of("doc2"),
                                "C",
                                List.of("doc3"),
                                "A,B",
                                List.of("doc4")));
        assertEquals(List.of("doc1", "doc2"), apply(metadata, "category", List.of("A", "B"), "in"));
        assertEquals(List.of("doc3", "doc4"), apply(metadata, "category", List.of("A", "B"), "not in"));
        assertEquals(List.of("doc4"), apply(metadata, "category", List.of("A,B"), "in"));
        assertEquals(List.of("doc1", "doc2"), apply(metadata, "category", " A , , B, ", "in"));
        assertEquals(List.of(), apply(metadata, "category", 123, "not in"));
        assertEquals(List.of(), apply(metadata, "category", Arrays.asList("A", null), "not in"));
        assertEquals(
                List.of(), apply(metadata, "category", List.of("A", Map.of("value", "B")), "not in"));
        assertEquals(List.of(), apply(metadata, "category", List.of("a"), "in"));
        assertEquals(
                List.of("doc2", "doc3", "doc4"), apply(metadata, "category", List.of("a"), "not in"));
    }

    @Test
    void originalNegativeOperatorsAreExistentialForMultivalueDocuments() {
        var metadata = Map.of("tags", Map.of("a", List.of("doc1"), "b", List.of("doc1")));
        assertEquals(List.of("doc1"), apply(metadata, "tags", "a", "≠"));
        assertEquals(List.of("doc1"), apply(metadata, "tags", "a", "not in"));
    }

    @Test
    void originalNumericAndDateComparisons() {
        var metadata =
                Map.of(
                        "score", Map.of("85", List.of("doc1"), "80", List.of("doc2"), "70", List.of("doc3")));
        assertEquals(List.of("doc1"), apply(metadata, "score", "80", ">"));
        assertEquals(List.of("doc3"), apply(metadata, "score", "80", "<"));
        assertEquals(List.of("doc1", "doc2"), apply(metadata, "score", 80.0, ">="));
        assertEquals(List.of("doc2", "doc3"), apply(metadata, "score", "80", "<="));
        assertTrue(compareValues("10", "2", ">"));
        assertTrue(compareValues("1_000", "20", ">"));
        assertTrue(compareValues("0x1p4", "2", ">"));
        assertFalse(compareValues("10 ", "2", ">"));
        assertTrue(compareValues("Banana", "apple", ">"));
        assertFalse(compareValues("xyz", "2024-01-01", ">"));
        assertTrue(compareValues("xyz", "2024-01-01", "≠"));
        assertTrue(compareValues("2024-99-99", "2024-01-01", ">"));
        var dates =
                Map.of(
                        "date",
                        Map.of(
                                "2024-06-01",
                                List.of("doc1"),
                                "2024-07-15",
                                List.of("doc2"),
                                "2024-05-01",
                                List.of("doc3")));
        assertEquals(List.of("doc2"), apply(dates, "date", "2024-06-01", ">"));
    }

    @Test
    void emptyAndMissingFieldsRemainDistinct() {
        var metadata = Map.of("status", Map.of("", List.of("doc1"), "active", List.of("doc2")));
        assertEquals(List.of("doc1"), apply(metadata, "status", "", "empty"));
        assertEquals(List.of("doc2"), apply(metadata, "status", "", "not empty"));
        assertEquals(List.of(), apply(metadata, "missing", "", "empty"));
        var conditions =
                List.of(new Condition("missing", "", "="), new Condition("status", "active", "="));
        assertEquals(List.of(), applyMetaFilter(metadata, conditions, "and"));
        assertEquals(List.of("doc2"), applyMetaFilter(metadata, conditions, "or"));
        assertEquals(List.of("doc2"), applyMetaFilter(metadata, conditions, "AND"));
    }

    @Test
    void manualNoMatchSentinelAndBaseScopeOrder() {
        List<String> base = List.of("doc-1", "doc-2");
        assertEquals(
                List.of(NO_MATCH_DOC_ID),
                applyMetaDataFilter(manual("author", "nobody", "="), AUTHORS, "", null, base));
        assertEquals(
                List.of("doc-1"),
                applyMetaDataFilter(manual("author", "Zhang San", "="), AUTHORS, "", null, base));
        assertEquals(
                List.of(NO_MATCH_DOC_ID),
                applyMetaDataFilter(manual("author", "Li Si", "="), AUTHORS, "", null, base));
        assertEquals(
                List.of("b", "a"), constrainDocIds(List.of("b", "a", "b", "c"), List.of("a", "b", "a")));
        assertSame(
                base,
                applyMetaDataFilter(
                        Map.of("method", "manual", "manual", List.of()), AUTHORS, "", null, base));
    }

    @Test
    void generatedFailureEmptyAndOutOfScopeKeepBaseScope() {
        List<String> base = List.of("doc-1", "doc-2");
        for (String answer :
                List.of(
                        "{\"conditions\":[],\"logic\":\"and\"}",
                        "{\"conditions\":[{\"key\":\"author\",\"op\":\"=\",\"value\":\"Li Si\"}]}",
                        "invalid output")) {
            ChatPipeline.ChatModel model = (messages, generation) -> new ChatPipeline.ModelAnswer(answer);
            assertNull(
                    applyMetaDataFilter(Map.of("method", "auto"), AUTHORS, "who wrote it?", model, base));
            assertNull(
                    applyMetaDataFilter(
                            Map.of("method", "semi_auto", "semi_auto", List.of("author")),
                            AUTHORS,
                            "who wrote it?",
                            model,
                            base));
        }
        ChatPipeline.ChatModel failure =
                (messages, generation) -> {
                    throw new Exception("model unavailable");
                };
        assertSame(
                base, applyMetaDataFilter(Map.of("method", "auto"), AUTHORS, "question", failure, base));
    }

    @Test
    void generatedFiltersCleanupAndPromptMatchSource() throws Exception {
        var generated =
                generate(
                        AUTHORS,
                        "Find Zhang San's documents",
                        null,
                        (messages, generation) -> {
                            assertEquals("Generate filters:", messages.getLast().content());
                            assertTrue(messages.getFirst().content().contains("2026-10-13"));
                            assertTrue(messages.getFirst().content().contains("Zhang San"));
                            assertFalse(messages.getFirst().content().contains("{{"));
                            assertTrue(generation.isEmpty());
                            return new ChatPipeline.ModelAnswer(
                                    "<think>reason</think>noise</think>```json\n"
                                            + "{\"conditions\":[{\"key\":\"author\",\"value\":\"Zhang"
                                            + " San\",\"op\":\"=\"}],\"logic\":\"and\"}\n"
                                            + "```");
                        },
                        "2026-10-13");
        assertEquals(
                List.of("doc-1"), applyMetaFilter(AUTHORS, generated.conditions(), generated.logic()));
        String prompt = renderPrompt("2026-10-13", "{}", "question", "{\"author\":\"=\"}");
        assertTrue(prompt.contains("- Operator constraints: {{ constraints }}"));
        var repaired =
                generate(
                        AUTHORS,
                        "question",
                        null,
                        (messages, generation) ->
                                new ChatPipeline.ModelAnswer(
                                        "{conditions:[{key:'author',value:'Zhang San',op:'='},],logic:'and',}"),
                        "2026-10-13");
        assertEquals(
                List.of("doc-1"), applyMetaFilter(AUTHORS, repaired.conditions(), repaired.logic()));
    }

    @Test
    void catalogFlatteningMatchesStoredMetadataShapes() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("author", "Alice");
        metadata.put("empty", "");
        metadata.put("nil", null);
        metadata.put("bool", true);
        metadata.put("year", 2024);
        metadata.put("tags", List.of("red,blue", "red", " green；white"));
        metadata.put("scalar", "red,blue");
        var flattened = flatten(List.of(Map.of("internalDocId", "a", "metadata", metadata)));
        assertEquals(Map.of("Alice", List.of("a")), flattened.get("author"));
        assertEquals(Map.of("2024", List.of("a")), flattened.get("year"));
        assertEquals(Set.of("red", "blue", "green", "white"), flattened.get("tags").keySet());
        assertEquals(Set.of("red,blue"), flattened.get("scalar").keySet());
        assertTrue(flattened.get("empty").isEmpty());
        assertTrue(flattened.get("bool").isEmpty());
        assertFalse(flattened.containsKey("nil"));
    }

    @Test
    void requestAliasesAndWrapperPreserveUnnarrowedScope() {
        var filter = Map.<String, Object>of("method", "auto");
        assertEquals(filter, configFromRequest(Map.of("metadataFilter", filter)));
        assertEquals(filter, configFromRequest(Map.of("meta_data_filter", filter)));
        List<String> base = List.of("a");
        var rows =
                List.of(
                        Map.<String, Object>of("internalDocId", "a", "metadata", Map.of("author", "Alice")));
        assertSame(
                base,
                filterDocumentIds(
                        filter,
                        rows,
                        "question",
                        (messages, generation) -> new ChatPipeline.ModelAnswer("{}"),
                        base));
        assertEquals(
                List.of(NO_MATCH_DOC_ID),
                filterDocumentIds(manual("author", "Bob", "="), rows, "question", null, base));
    }

    @Test
    void exactUpstreamRepairVectorsBeyondLenientJsonParsing() throws Exception {
        // kaptinlin/jsonrepair v0.4.8 repair_test.go 的逐字输入/输出。
        assertEquals("{\"a\":2}", repairJson("{\"a\":2"));
        assertEquals("[[1,2,3]]", repairJson("[[1,2,3,"));
        assertEquals("{\"array\": [{},{}]}", repairJson("{\"array\": [{}{}]}"));
        assertEquals("{\"a\":2,\n\"b\":3\n}", repairJson("{\"a\":2\n\"b\":3\n}"));
        assertEquals("{\"a\": \"b\"}", repairJson("{\"a\" \"b\"}"));
        assertEquals("{\"a\":2e-0}", repairJson("{\"a\":2e-}"));
        assertThrows(java.io.IOException.class, () -> repairJson(""));
    }

    @Test
    void truncatedGeneratedMetadataUsesExactRepairHelper() throws Exception {
        Generated repaired =
                generate(
                        AUTHORS,
                        "question",
                        null,
                        (messages, generation) ->
                                new ChatPipeline.ModelAnswer(
                                        "{\"conditions\":[{\"key\":\"author\",\"value\":\"Zhang San\",\"op\":\"=\"}"),
                        "2026-10-13");
        assertEquals(
                List.of("doc-1"), applyMetaFilter(AUTHORS, repaired.conditions(), repaired.logic()));
    }

    @Test
    void generatedStructDecodingPreservesNullEntriesAndRejectsTrailingProse() throws Exception {
        Generated generated =
                generate(
                        AUTHORS,
                        "question",
                        null,
                        (messages, generation) ->
                                new ChatPipeline.ModelAnswer(
                                        "{\"Conditions\":[null,{\"Key\":\"author\",\"Value\":\"Zhang"
                                                + " San\",\"Op\":\"=\"}],\"Logic\":\"or\"}"),
                        "2026-10-13");
        assertEquals(2, generated.conditions().size());
        assertEquals(
                List.of("doc-1"), applyMetaFilter(AUTHORS, generated.conditions(), generated.logic()));
        Generated invalid =
                generate(
                        AUTHORS,
                        "question",
                        null,
                        (messages, generation) ->
                                new ChatPipeline.ModelAnswer(
                                        "{\"conditions\":[{\"key\":\"author\",\"value\":\"Zhang San\",\"op\":\"=\"}]}"
                                                + " trailing explanation"),
                        "2026-10-13");
        assertTrue(invalid.conditions().isEmpty());
    }
}
