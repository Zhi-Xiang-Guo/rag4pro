package ai.ragflow4j.ragflow;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SynonymTest {
    @TempDir
    Path temporary;

    @Test
    void customDictionaryTakesPrecedenceAndPreservesOrder() throws Exception {
        Path dictionary = temporary.resolve("synonym.json");
        Files.writeString(dictionary, "{\"DOG\":[\"first\",\"second\",17],\"A B\":\"joined\"}");
        var synonym = new Synonym(dictionary, NativeTokenizer.defaultResourcePath().resolve("wordnet"));
        assertEquals(List.of("first"), synonym.lookup(" DOG ", 1));
        assertEquals(List.of("joined"), synonym.lookup(" a\t  b ", 8));
        assertEquals(List.of("first", "second"), synonym.lookup("dog", 0));
    }

    @Test
    void wordnetFallbackMatchesGoOracle() throws Exception {
        Path dictionary = temporary.resolve("synonym.json");
        Files.writeString(dictionary, "{}");
        var synonym = new Synonym(dictionary, NativeTokenizer.defaultResourcePath().resolve("wordnet"));
        assertEquals(
                List.of("andiron", "cad", "chase", "frank", "frump", "pawl"), synonym.lookup("dog", 8));
        assertEquals(List.of("child"), synonym.lookup("children", 8));
        assertEquals(List.of("mouse", "shiner"), synonym.lookup("mice", 8));
        assertEquals(List.of("cable car"), synonym.lookup("car", 8));
        assertEquals(
                List.of("campaign", "carry", "function", "guide", "hunt", "ladder", "linear", "melt"),
                synonym.lookup("running", 8));
        assertEquals(List.of(), synonym.lookup("Dog", 8));
        assertEquals(List.of(), synonym.lookup(" dog ", 8));
        assertEquals(List.of(), synonym.lookup("", 8));
    }
}
