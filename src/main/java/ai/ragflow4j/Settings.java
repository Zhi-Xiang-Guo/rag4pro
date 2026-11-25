package ai.ragflow4j;

/** Runtime settings with a fallback for existing local installations. */
public final class Settings {
    private Settings() {}

    public static String get(String name, String fallback) {
        String value = System.getenv("EVIDENCE_RAG_" + name);
        if (value == null) value = System.getenv("RAGFLOW4J_" + name);
        return value == null || value.isBlank() ? fallback : value;
    }
}
