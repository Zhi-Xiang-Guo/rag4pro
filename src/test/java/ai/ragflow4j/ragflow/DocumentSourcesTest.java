package ai.ragflow4j.ragflow;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

class DocumentSourcesTest {
    @TempDir Path directory;

    @Test void revisionsKeepTheirOwnOriginalAndAreScopedToTheDocument() throws Exception {
        var store = new DocumentSources(directory);
        var first = UUID.randomUUID().toString();
        var second = UUID.randomUUID().toString();
        var body = "旧文档内容".getBytes(StandardCharsets.UTF_8);
        var pages = List.of(new DocumentSources.Unit("page", 2, "旧文档内容"));
        var version = store.save("knowledge", "document", first, "guide.pdf", "pdf", body, pages);
        store.save("knowledge", "document", second, "guide.pdf", "pdf", new byte[]{1},
                List.of(new DocumentSources.Unit("page", 1, "新版")));
        var reopened = new DocumentSources(directory);
        assertEquals(version, reopened.find("knowledge", "document", first));
        assertArrayEquals(body, reopened.original("knowledge", "document", first));
        assertNull(reopened.find("other", "document", first));
        assertNull(reopened.find("knowledge", "another", first));
        assertThrows(IllegalArgumentException.class, () -> reopened.find("knowledge", "document", "../old"));
        assertThrows(IllegalStateException.class, () ->
                store.save("knowledge", "document", first, "guide.pdf", "pdf", new byte[]{2}, pages));
    }
}
