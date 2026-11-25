package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class Cl100kTrimTest {
    private final Cl100k tokens = new Cl100k();

    @Test
    void partialUtf8ScalarIsClampedLikeOriginalGoDecoder() {
        assertEquals(2, tokens.count("😀"));
        assertEquals("", tokens.trim("😀", 1));
        assertEquals("😀", tokens.trim("😀", 2));
        assertEquals("", tokens.trim("text", -1));
    }

    @Test
    void unicodeCutsAreValidPrefixesWithinBudget() {
        for (String text :
                new String[]{
                        "中文检索增强生成😀终", "مرحبا بالعالم！", "á café 🧑‍💻\n끝", "literal � replacement"
                }) {
            for (int limit = 0; limit <= tokens.count(text); limit++) {
                String trimmed = tokens.trim(text, limit);
                assertTrue(text.startsWith(trimmed), () -> "Non-prefix " + trimmed);
                assertTrue(tokens.count(trimmed) <= limit);
            }
        }
    }
}
