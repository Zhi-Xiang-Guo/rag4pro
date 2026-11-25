package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;

/**
 * 对照 internal/service/nlp/synonym.go，从同义词资源及 WordNet 扩展查询词；原版 Redis 重载未启用。
 */
public final class Synonym {
    private final Map<String, List<String>> dictionary = new HashMap<>();
    private final WordNet wordNet;

    public Synonym() {
        this(NativeTokenizer.defaultResourcePath());
    }

    public Synonym(Path resourceRoot) {
        this(resourceRoot.resolve("synonym.json"), resourceRoot.resolve("wordnet"));
    }

    public Synonym(Path dictionaryPath, Path wordnetDirectory) {
        WordNet loaded = null;
        if (wordnetDirectory != null) {
            try {
                loaded = new WordNet(wordnetDirectory);
            } catch (IOException exception) {
                System.getLogger(Synonym.class.getName())
                        .log(
                                System.Logger.Level.WARNING,
                                "WordNet unavailable at " + wordnetDirectory,
                                exception);
            }
        }
        wordNet = loaded;
        try {
            JsonNode root = new ObjectMapper().readTree(dictionaryPath.toFile());
            root.fields()
                    .forEachRemaining(
                            entry -> {
                                List<String> values = new ArrayList<>();
                                JsonNode value = entry.getValue();
                                if (value.isTextual()) values.add(value.asText());
                                else if (value.isArray()) {
                                    for (JsonNode item : value) if (item.isTextual()) values.add(item.asText());
                                } else return;
                                dictionary.put(entry.getKey().toLowerCase(Locale.ROOT), List.copyOf(values));
                            });
        } catch (IOException exception) {
            System.getLogger(Synonym.class.getName())
                    .log(
                            System.Logger.Level.WARNING,
                            "Synonym dictionary unavailable at " + dictionaryPath,
                            exception);
        }
    }

    public List<String> lookup(String token, int topN) {
        if (token.isEmpty()) return new ArrayList<>();
        int limit = topN <= 0 ? 8 : topN;
        String key =
                token.replaceAll("(?U)^\\s+|\\s+$", "").replaceAll("[ \\t]+", " ").toLowerCase(Locale.ROOT);
        List<String> custom = dictionary.get(key);
        if (custom != null) return new ArrayList<>(custom.subList(0, Math.min(limit, custom.size())));
        // Go checks the original token, preserving the lowercase-only fallback contract.
        if (!token.matches("[a-z]+") || wordNet == null) return new ArrayList<>();
        TreeSet<String> words = new TreeSet<>();
        for (WordNet.Synset synset : wordNet.synsets(token, "")) {
            words.add(synset.name().split("\\.", 2)[0].replace('_', ' '));
        }
        words.remove(token);
        words.remove("");
        return new ArrayList<>(words.stream().limit(limit).toList());
    }
}
