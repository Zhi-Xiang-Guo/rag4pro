package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.UUID;

/** Original files and search text belong to an immutable document revision. */
public final class DocumentSources {
    public record Unit(String kind, int number, String text) {}
    public record Version(String revision, String name, String format, String sha256, List<Unit> units) {
        public Version {
            units = List.copyOf(units);
        }
    }

    private static final ObjectMapper JSON = new ObjectMapper();
    private final Path root;

    public DocumentSources(Path root) {
        this.root = root;
    }

    public Version save(String dataset, String document, String revision, String name,
                        String format, byte[] original, List<Unit> units) throws IOException {
        Path target = path(dataset, document, revision);
        String digest = DocumentCatalog.sourceDigest(original);
        Version existing = find(dataset, document, revision);
        if (existing != null) {
            if (!digest.equals(existing.sha256())) throw new IllegalStateException("文档版本已绑定其他文件");
            return existing;
        }
        Version version = new Version(revision, name, format, digest, units);
        Files.createDirectories(target.getParent());
        Path staged = Files.createTempDirectory(target.getParent(), ".source-");
        try {
            Files.write(staged.resolve("original"), original);
            JSON.writeValue(staged.resolve("version.json").toFile(), version);
            Files.move(staged, target, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            Files.deleteIfExists(staged.resolve("original"));
            Files.deleteIfExists(staged.resolve("version.json"));
            Files.deleteIfExists(staged);
        }
        return version;
    }

    public Version find(String dataset, String document, String revision) throws IOException {
        Path manifest = path(dataset, document, revision).resolve("version.json");
        return Files.isRegularFile(manifest) ? JSON.readValue(manifest.toFile(), Version.class) : null;
    }

    public byte[] original(String dataset, String document, String revision) throws IOException {
        return Files.readAllBytes(path(dataset, document, revision).resolve("original"));
    }

    public static String contentType(String format) {
        return switch (format) {
            case "pdf" -> "application/pdf";
            case "docx" -> "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
            default -> "text/plain; charset=utf-8";
        };
    }

    private Path path(String dataset, String document, String revision) {
        String validated = UUID.fromString(revision).toString();
        return root.resolve(DocumentCatalog.identity(dataset, document)).resolve(validated);
    }
}
