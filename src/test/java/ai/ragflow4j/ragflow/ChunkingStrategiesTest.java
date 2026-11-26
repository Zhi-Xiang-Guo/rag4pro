package ai.ragflow4j.ragflow;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ChunkingStrategiesTest {
    private final Cl100k tokens = new Cl100k();
    private final ChunkingStrategies strategies = new ChunkingStrategies(tokens);
    private final GeneralChunker.Config config = new GeneralChunker.Config(40, List.of(), List.of(), 0);

    @Test
    void fixedWindowsPreserveUnicodeAtTokenBoundaries() {
        String input = "每次回退后记录事件编号🙂。".repeat(60);
        var pieces = strategies.plan(input, "txt", "fixed", config).pieces();
        assertTrue(pieces.size() > 1);
        StringBuilder restored = new StringBuilder();
        for (var fragment : pieces) {
            String text = (String) fragment.get("text");
            assertTrue(tokens.count(text) <= 40);
            assertFalse(text.contains("\uFFFD"));
            restored.append(text);
        }
        assertEquals(input, restored.toString());
    }

    @Test
    void nestedSectionsContainMultipleParentsAndRespectHeadingBoundaries() {
        String section = "An operator checks the payment receipt and the completion timestamp before retrying.\n\n".repeat(23);
        String markdown = "# Runbook\n\n## Payment\n\n" + section + "## Refunding\n\nRefunds require approval.";
        var plan = strategies.plan(markdown, "md", "hierarchical", config);
        var first = plan.pieces().getFirst();
        assertEquals("Runbook / Payment", first.get("section_kwd"));
        assertTrue(first.get("section_context").toString().length() > first.get("mom").toString().length());
        assertFalse(first.get("section_context").toString().contains("Refunds require approval"));
        assertEquals("Runbook / Refunding", plan.pieces().getLast().get("section_kwd"));
        assertTrue(plan.pieces().stream().allMatch(c -> tokens.count(c.get("text").toString()) <= 40));
    }

    @Test
    void catalogRecognizesExistingCorporaAndPersistsAnEmptyKnowledgeBase(@TempDir Path folder) throws Exception {
        var path = folder.resolve("documents.json");
        var catalog = new DocumentCatalog(path);
        catalog.put("old", "doc", Map.of("title", "Existing"));
        catalog.putDataset("new", Map.of("name", "New", "chunkStrategy", "parent_child"));
        var reopened = new DocumentCatalog(path);
        assertEquals("New", reopened.dataset("new").get("name"));
        assertEquals(List.of("new", "old"), reopened.datasets().stream().map(d -> d.get("datasetId")).toList());
        assertThrows(IllegalArgumentException.class, () -> reopened.deleteEmptyDataset("old"));
        reopened.deleteEmptyDataset("new");
        assertEquals(List.of("old"), new DocumentCatalog(path).datasets().stream().map(d -> d.get("datasetId")).toList());
    }

    @Test
    void diversityKeepsTheFirstHitFromEachDocumentAheadOfDuplicates() {
        var chunks = Map.<String, Map<String, Object>>of("a0", Map.of("doc_id", "a"), "a1", Map.of("doc_id", "a"),
                "b0", Map.of("doc_id", "b"), "c0", Map.of("doc_id", "c"), "b1", Map.of("doc_id", "b"));
        assertEquals(List.of("a0", "b0", "c0", "a1", "b1"),
                StandardRetrieval.preferDifferentDocuments(List.of("a0", "a1", "b0", "b1", "c0"), chunks));
    }
}
