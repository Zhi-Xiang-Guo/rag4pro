// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations.
package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import ai.ragflow4j.ragflow.ContextBuilder.Message;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

/**
 * Fixtures mirror Go kbPrompt/messageFitIn/citation branches; deterministic token budgets are
 * injected.
 */
class ChatPipelineTest {
    private static final ContextBuilder.TokenCounter TOKENS =
            new ContextBuilder.TokenCounter() {
                public int count(String text) {
                    return text.codePointCount(0, text.length());
                }

                public String trim(String text, int limit) {
                    return text.substring(
                            0, text.offsetByCodePoints(0, Math.min(Math.max(0, limit), count(text))));
                }
            };

    private static Map<String, Object> chunk(String id, String doc, String body) {
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("chunk_id", id);
        chunk.put("doc_id", doc);
        chunk.put("kb_id", "kb");
        chunk.put("docnm_kwd", "Title\n\nline");
        chunk.put("content_with_weight", body);
        chunk.put("vector", new double[]{1, 0});
        return chunk;
    }

    private static ChatPipeline.Request request(
            boolean quote, String empty, Map<String, Object> generation) {
        return new ChatPipeline.Request(
                List.of(new Message("user", "question")),
                "Answer from {knowledge}",
                null,
                Map.of(),
                generation,
                30000,
                empty,
                quote,
                true,
                "",
                false);
    }

    @Test
    void knowledgeKeepsStablePoolIdsMetadataOrderAndBoundaryCrossingChunk() {
        Map<String, Object> first = chunk("a", "d", "1234567890");
        first.put("document_metadata", Map.of("z", "last", "a", "first"));
        List<String> blocks =
                new ContextBuilder(TOKENS).knowledge(List.of(first, chunk("b", "d", "two")), 10);
        assertEquals(
                List.of(
                        "\nID: 0\n├── Title: Title line\n├── a: first\n├── z: last\n└── Content:\n1234567890"),
                blocks);
    }

    @Test
    void knowledgePreservesUpstreamTwoPassBehaviorForEmptyContent() {
        assertTrue(
                new ContextBuilder(TOKENS)
                        .knowledge(List.of(chunk("empty", "d", ""), chunk("b", "d", "text")), 100)
                        .isEmpty());
    }

    @Test
    void messageFitDiscardsHistoryBeforeTrimmingAndKeepsLastFirstWhenSystemDominates() {
        var result =
                new ContextBuilder(TOKENS)
                        .fitMessages(
                                List.of(
                                        new Message("system", "01234567890123456789"),
                                        new Message("user", "old"),
                                        new Message("assistant", "history"),
                                        new Message("user", "new")),
                                10);
        assertEquals(
                List.of(new Message("system", "0123456"), new Message("user", "new")), result.messages());
        assertEquals(10, result.tokens());
        var other =
                new ContextBuilder(TOKENS)
                        .fitMessages(
                                List.of(new Message("system", "system"), new Message("user", "0123456789")), 10);
        assertEquals("0123", other.messages().getLast().content());
    }

    @Test
    void templateAndCitationPromptPreserveUpstreamRules() {
        assertEquals(
                "x   {unknown}",
                ContextBuilder.formatPrompt("{who} {knowledge} {unknown}", Map.of("who", "x")));
        String prompt = ContextBuilder.citationPrompt();
        assertTrue(prompt.contains("## Example 6: RTL Language (Arabic)"));
        assertTrue(prompt.endsWith("NEVER merge consecutive citations into a range such as [ID:i-j]."));
    }

    @Test
    void citationFormatsArabicDigitsAndBareYears() {
        assertEquals("[ID:12] [ID:3] [ID:4]", CitationService.repair("(ID: ١٢) 【ID:۳】 (**ID:4**)"));
        assertEquals(List.of(1, 0), CitationService.extract("[ID:1] [٠] [ID:1] [99]", 2));
        assertEquals("Year [2024] ", CitationService.strip("Year [2024] [ID:1]"));
    }

    @Test
    void citationInsertionCapsFourAndDeduplicatesAcrossSentences() throws Exception {
        List<double[]> vectors = Collections.nCopies(6, new double[]{1, 0});
        var result =
                CitationService.insert(
                        "First fact. Second fact.",
                        vectors,
                        texts -> texts.stream().map(t -> new double[]{1, 0}).toList());
        assertEquals(List.of(0, 1, 2, 3), result.citedIndices());
        for (int i = 0; i < 4; i++)
            assertEquals(1, result.answer().split("\\[ID:" + i + "\\]", -1).length - 1);
        assertFalse(result.answer().contains("[ID:4]"));
    }

    @Test
    void cosineThresholdDescendsGloballyAndLeavesLowScoresUncited() throws Exception {
        var result =
                CitationService.insert(
                        "Some factual answer",
                        List.of(new double[]{0.4, Math.sqrt(0.84)}),
                        texts -> List.of(new double[]{1, 0}));
        assertEquals(List.of(0), result.citedIndices());
        assertTrue(
                CitationService.insert(
                                "Some factual answer",
                                List.of(new double[]{0.2, Math.sqrt(0.96)}),
                                texts -> List.of(new double[]{1, 0}))
                        .citedIndices()
                        .isEmpty());
    }

    @Test
    void referencesHonorAliasPresenceCitationDocsAndQuoteToggle() {
        Map<String, Object> first = chunk("a", "doc1", "first");
        first.put("content", "");
        List<Map<String, Object>> chunks = List.of(first, chunk("b", "doc2", "second"));
        List<Map<String, Object>> docs = List.of(Map.of("doc_id", "doc1"), Map.of("doc_id", "doc2"));
        var decorated = CitationService.decorate("Answer (ID:1)", chunks, docs, true, true, null, null);
        assertEquals(List.of(1), decorated.citedIndices());
        assertEquals(List.of(Map.of("doc_id", "doc2")), decorated.reference().get("doc_aggs"));
        var refs = (List<?>) decorated.reference().get("chunks");
        assertEquals("", ((Map<?, ?>) refs.getFirst()).get("content"));
        assertFalse(((Map<?, ?>) refs.getFirst()).containsKey("vector"));
        var hidden = CitationService.decorate("Answer [ID:0]", chunks, docs, true, false, null, null);
        assertEquals("Answer ", hidden.answer());
        assertFalse(hidden.reference().containsKey("doc_aggs"));
        assertEquals(2, ((List<?>) hidden.reference().get("chunks")).size());
    }

    @Test
    void hydrationRunsOnlyWhenModelDidNotSupplyMarkers() {
        AtomicInteger hydrations = new AtomicInteger();
        Map<String, Object> source = chunk("a", "d", "Some factual answer");
        source.remove("vector");
        CitationService.Embedder encoder =
                texts -> texts.stream().map(t -> new double[]{1, 0}).toList();
        CitationService.VectorHydrator hydrate =
                (chunks, dimension) -> {
                    hydrations.incrementAndGet();
                    assertEquals(2, dimension);
                    chunks.getFirst().put("vector", new double[]{1, 0});
                };
        var result =
                CitationService.decorate(
                        "Some factual answer", List.of(source), List.of(), true, true, encoder, hydrate);
        assertEquals(List.of(0), result.citedIndices());
        assertFalse(source.containsKey("vector"));
        CitationService.decorate(
                "Some answer [ID:0]", List.of(source), List.of(), true, true, encoder, hydrate);
        assertEquals(1, hydrations.get());
    }

    @Test
    void completeCallsModelWithKnowledgeAndReturnsCitations() throws Exception {
        var pipeline =
                new ChatPipeline(
                        (messages, generation) -> {
                            assertTrue(messages.getFirst().content().contains("ID: 0"));
                            assertTrue(messages.getFirst().content().contains("Citation Requirements"));
                            assertEquals(100, generation.get("max_tokens"));
                            return new ChatPipeline.ModelAnswer("The fact (ID:0)");
                        },
                        TOKENS,
                        null,
                        null);
        var result =
                pipeline.complete(
                        request(true, "empty", Map.of("max_tokens", 100)),
                        new ChatPipeline.Retrieval(
                                List.of(chunk("a", "d", "fact")), List.of(Map.of("doc_id", "d"))));
        assertEquals("The fact [ID:0]", result.answer());
        assertEquals(List.of(0), result.citedIndices());
        assertFalse(result.emptyResponse());
    }

    @Test
    void emptyResponseSkipsModelAndAttachmentsBypassIt() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        var pipeline =
                new ChatPipeline(
                        (messages, generation) -> {
                            calls.incrementAndGet();
                            return new ChatPipeline.ModelAnswer("answer");
                        },
                        TOKENS,
                        null,
                        null);
        var empty = new ChatPipeline.Retrieval(List.of(), List.of());
        assertTrue(pipeline.complete(request(false, "empty", Map.of()), empty).emptyResponse());
        assertEquals(0, calls.get());
        var r = request(false, "empty", Map.of());
        var attached =
                new ChatPipeline.Request(
                        r.messages(),
                        r.systemPrompt(),
                        r.parameters(),
                        r.variables(),
                        r.generation(),
                        r.modelMaxTokens(),
                        r.emptyResponse(),
                        r.quote(),
                        r.storeHistoryMessages(),
                        "attachment",
                        false);
        assertEquals("answer", pipeline.complete(attached, empty).answer());
        assertEquals(1, calls.get());
    }

    @Test
    void promptParametersValidateAndCitationBudgetClampsCompletion() {
        var pipeline = new ChatPipeline((m, g) -> new ChatPipeline.ModelAnswer(""), TOKENS, null, null);
        var request =
                new ChatPipeline.Request(
                        List.of(new Message("user", "question")),
                        "{knowledge} {required}",
                        List.of(new ChatPipeline.Parameter("required", false)),
                        Map.of(),
                        Map.of(),
                        30000,
                        "",
                        true,
                        true,
                        "",
                        false);
        assertThrows(
                IllegalArgumentException.class, () -> pipeline.prepare(request, List.of("knowledge")));
        var prepared =
                pipeline.prepare(request(true, "", Map.of("max_tokens", 50000)), List.of("knowledge"));
        assertEquals(30000 - prepared.promptTokens(), prepared.generation().get("max_tokens"));
        assertTrue(prepared.promptTokens() > ContextBuilder.citationPrompt().length());
    }
}
