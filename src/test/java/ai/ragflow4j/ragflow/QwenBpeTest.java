package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.*;

import org.junit.jupiter.api.Test;

class QwenBpeTest {
    private static final QwenBpe QWEN = new QwenBpe();

    @Test
    void tokenIdsMatchHuggingFaceOracleWithSourcePinnedAsset() throws Exception {
        try (var stream = getClass().getResourceAsStream("/ragflow/qwen_bpe_oracle.json")) {
            for (var fixture : new ObjectMapper().readTree(stream)) {
                List<Integer> expected = new ArrayList<>();
                fixture.get("ids").forEach(id -> expected.add(id.asInt()));
                assertEquals(expected, QWEN.encode(fixture.get("text").asText()));
            }
        }
    }

    @Test
    void trimPreservesUnicodePrefixAndActualQwenBudget() {
        String text = "中文测试😀مرحبا e\u0301 123456789 ".repeat(20);
        for (int limit : new int[]{0, 1, 5, 31, 64, 127}) {
            String trimmed = QWEN.trim(text, limit);
            assertTrue(text.startsWith(trimmed));
            assertTrue(QWEN.count(trimmed) <= limit);
            assertFalse(trimmed.endsWith("\ud83d"));
        }
    }
}
