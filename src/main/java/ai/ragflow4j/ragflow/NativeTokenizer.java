package ai.ragflow4j.ragflow;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/**
 * 通过 JNA 调用原版 RAGFlow C++ 词法分析器，输出供词法检索使用的粗细粒度词序列。
 *
 * <p>每个实例持有一个原生句柄，使用结束后必须 close；同步方法保护句柄及可变语言状态。 这里的词序列与 Cl100k/QwenBpe 用来计数和裁剪的 BPE token 用途不同。
 */
public final class NativeTokenizer implements AutoCloseable {
    private interface AnalyzerLibrary extends Library {
        Pointer RAGAnalyzer_Create(String resourcePath);

        int RAGAnalyzer_Load(Pointer handle);

        void RAGAnalyzer_Destroy(Pointer handle);

        void RAGAnalyzer_SetLanguage(Pointer handle, String language);

        Pointer RAGAnalyzer_Tokenize(Pointer handle, String text);

        Pointer RAGAnalyzer_FineGrainedTokenize(Pointer handle, String tokens);

        int RAGAnalyzer_GetTermFreq(Pointer handle, String term);

        Pointer RAGAnalyzer_GetTermTag(Pointer handle, String term);

        void RAGFlow_FreeString(Pointer value);
    }

    private final AnalyzerLibrary library;
    private Pointer handle;

    public NativeTokenizer() {
        this(defaultLibraryPath(), defaultResourcePath());
    }

    public NativeTokenizer(Path libraryPath, Path resourcePath) {
        Path absoluteLibrary = libraryPath.toAbsolutePath();
        Path absoluteResources = resourcePath.toAbsolutePath();
        if (!Files.isRegularFile(absoluteLibrary)) {
            throw new IllegalStateException(
                    "RAGFlow native analyzer missing at "
                            + absoluteLibrary
                            + ". Run scripts/native-build.sh first.");
        }
        requireResource(absoluteResources, "rag/huqie.txt");
        requireResource(absoluteResources, "rag/pos-id.def");
        for (String part : new String[]{"noun", "verb", "adj", "adv"}) {
            requireResource(absoluteResources, "wordnet/index." + part);
            requireResource(absoluteResources, "wordnet/" + part + ".exc");
        }
        library =
                Native.load(
                        absoluteLibrary.toString(),
                        AnalyzerLibrary.class,
                        Map.of(Library.OPTION_STRING_ENCODING, StandardCharsets.UTF_8.name()));
        handle = library.RAGAnalyzer_Create(absoluteResources.toString());
        if (handle == null) throw new IllegalStateException("Cannot create RAGFlow native analyzer");
        int result = library.RAGAnalyzer_Load(handle);
        if (result != 0) {
            close();
            throw new IllegalStateException(
                    "Cannot load RAGFlow dictionaries at " + absoluteResources + ", native status " + result);
        }
    }

    public static Path defaultLibraryPath() {
        String extension = System.getProperty("os.name").startsWith("Mac") ? "dylib" : "so";
        return Path.of(
                setting("RAGFLOW_NATIVE_LIBRARY", "native/build/libragflow_tokenizer." + extension));
    }

    public static Path defaultResourcePath() {
        return Path.of(setting("RAGFLOW_DICT_PATH", "native/resources"));
    }

    public synchronized String tokenize(String text) {
        return tokenize(text, "English");
    }

    public synchronized String tokenize(String text, String language) {
        selectLanguage(language);
        return readString(library.RAGAnalyzer_Tokenize(handle, cString(text)), false);
    }

    public synchronized String fineGrainedTokenize(String tokens) {
        return fineGrainedTokenize(tokens, "English");
    }

    /**
     * 对粗分词结果继续拆细，补充复合词等粒度；通常传入 tokenize 的返回值。
     */
    public synchronized String fineGrainedTokenize(String tokens, String language) {
        selectLanguage(language);
        return readString(library.RAGAnalyzer_FineGrainedTokenize(handle, cString(tokens)), false);
    }

    public synchronized int frequency(String term) {
        requireOpen();
        return library.RAGAnalyzer_GetTermFreq(handle, cString(term));
    }

    public int freq(String term) {
        return frequency(term);
    }

    public synchronized String tag(String term) {
        requireOpen();
        return readString(library.RAGAnalyzer_GetTermTag(handle, cString(term)), true);
    }

    private void selectLanguage(String language) {
        requireOpen();
        // 每次请求先重置可变词干器（含变音符折叠状态），避免前一请求的语言影响当前请求。
        library.RAGAnalyzer_SetLanguage(handle, "English");
        if (language != null && !language.isBlank()) {
            library.RAGAnalyzer_SetLanguage(handle, cString(language));
        }
    }

    private String readString(Pointer result, boolean allowMissing) {
        if (result == null) {
            if (allowMissing) return "";
            throw new IllegalStateException("RAGFlow native tokenization failed");
        }
        try {
            return result.getString(0, StandardCharsets.UTF_8.name());
        } finally {
            library.RAGFlow_FreeString(result);
        }
    }

    private void requireOpen() {
        if (handle == null) throw new IllegalStateException("RAGFlow native tokenizer is closed");
    }

    private static String cString(String text) {
        Objects.requireNonNull(text, "text");
        if (text.indexOf('\0') >= 0)
            throw new IllegalArgumentException("Native tokenizer input contains NUL");
        return text;
    }

    private static String setting(String name, String fallback) {
        String value = System.getProperty(name, System.getenv(name));
        return value == null || value.isBlank() ? fallback : value;
    }

    private static void requireResource(Path root, String relative) {
        if (!Files.isRegularFile(root.resolve(relative))) {
            throw new IllegalStateException(
                    "RAGFlow dictionary resource missing at " + root.resolve(relative));
        }
    }

    @Override
    public synchronized void close() {
        if (handle != null) {
            library.RAGAnalyzer_Destroy(handle);
            handle = null;
        }
    }
}
