package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import ai.ragflow4j.ragflow.ContextBuilder.Message;

import java.time.*;
import java.util.*;

import org.junit.jupiter.api.Test;

class QueryRefinerTest {
    @Test
    void fullQuestionRendersExactDatesConversationAndLanguage() {
        Clock clock = Clock.fixed(Instant.parse("2026-11-09T18:00:00Z"), ZoneId.of("Asia/Shanghai"));
        var refiner =
                new QueryRefiner(
                        (messages, generation) -> {
                            String prompt = messages.getFirst().content();
                            assertTrue(prompt.contains("based on today (2026-11-10)"));
                            assertTrue(prompt.contains("\"yesterday\" = 2026-11-09, \"tomorrow\" = 2026-11-11"));
                            assertTrue(
                                    prompt.contains(
                                            "- DON'T generate anything except a refined question.\n"
                                                    + "- Text generated MUST be in Chinese."));
                            assertTrue(prompt.endsWith("USER: Who?\nASSISTANT: Alice\nUSER: And tomorrow?\n"));
                            assertEquals("Output: ", messages.getLast().content());
                            assertTrue(generation.isEmpty());
                            return new ChatPipeline.ModelAnswer(
                                    "<think>first</think><think>second</think> Resolved question ");
                        },
                        clock);
        assertEquals(
                "Resolved question",
                refiner.fullQuestion(
                        List.of(
                                new Message("system", "ignored"),
                                new Message("user", "Who?"),
                                new Message("assistant", "Alice"),
                                new Message("user", "And tomorrow?")),
                        "Chinese"));
    }

    @Test
    void fullQuestionFallbackRetainsLatestQuestionAndError() {
        var messages = List.of(new Message("user", "original"));
        var failed =
                new QueryRefiner(
                        (m, g) -> {
                            throw new IllegalStateException("provider");
                        });
        var outcome = failed.fullQuestionOutcome(messages, "");
        assertEquals("original", outcome.text());
        assertNotNull(outcome.error());
        assertEquals(
                "original",
                new QueryRefiner((m, g) -> new ChatPipeline.ModelAnswer("**ERROR** failure"))
                        .fullQuestion(messages, ""));
        assertEquals(
                "original",
                new QueryRefiner((m, g) -> new ChatPipeline.ModelAnswer(" ")).fullQuestion(messages, ""));
    }

    @Test
    void keywordExtractionUsesOriginalTemplateAndCommaJoin() throws Exception {
        var refiner =
                new QueryRefiner(
                        (messages, generation) -> {
                            assertTrue(
                                    messages.getFirst().content().contains("top 3 important keywords/phrases"));
                            assertTrue(messages.getFirst().content().endsWith("content\n"));
                            assertEquals(0.2, generation.get("temperature"));
                            return new ChatPipeline.ModelAnswer("<think>x</think>alpha,beta ");
                        });
        assertEquals("alpha,beta", refiner.keywords("content", 0));
        assertEquals("question,alpha,beta", QueryRefiner.appendKeywords("question", "alpha,beta"));
        assertEquals("question", QueryRefiner.appendKeywords("question", ""));
    }

    @Test
    void crossLanguagesPreservesSourceResponseCleanup() {
        var refiner =
                new QueryRefiner(
                        (messages, generation) -> {
                            assertTrue(messages.getLast().content().contains("Chinese, French"));
                            assertEquals(0.2, generation.get("temperature"));
                            return new ChatPipeline.ModelAnswer("<think>x</think>Output: 你好\n===\nBonjour\n");
                        });
        assertEquals("你好\nBonjour", refiner.crossLanguages("Hello", List.of("Chinese", "French")));
        var failed =
                new QueryRefiner(
                        (m, g) -> {
                            throw new IllegalStateException("provider");
                        });
        assertEquals("Hello", failed.crossLanguages("Hello", List.of("Chinese")));
    }
}
