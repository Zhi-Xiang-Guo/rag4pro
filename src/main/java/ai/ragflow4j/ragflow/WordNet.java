package ai.ragflow4j.ragflow;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 对照 internal/service/nlp/wordnet.go，读取英文词汇资源，完成 Morphy 词形还原和同义词集选择。
 */
public final class WordNet {
    public record Synset(
            String name,
            String pos,
            int offset,
            List<String> lemmas,
            String definition,
            List<String> examples) {
    }

    private static final List<String> PARTS = List.of("n", "v", "a", "r");
    private static final Map<String, String> SUFFIX =
            Map.of("n", "noun", "v", "verb", "a", "adj", "r", "adv");
    private static final Map<String, String[][]> RULES =
            Map.of(
                    "n",
                    new String[][]{
                            {"s", ""},
                            {"ses", "s"},
                            {"ves", "f"},
                            {"xes", "x"},
                            {"zes", "z"},
                            {"ches", "ch"},
                            {"shes", "sh"},
                            {"men", "man"},
                            {"ies", "y"}
                    },
                    "v",
                    new String[][]{
                            {"s", ""},
                            {"ies", "y"},
                            {"es", "e"},
                            {"es", ""},
                            {"ed", "e"},
                            {"ed", ""},
                            {"ing", "e"},
                            {"ing", ""}
                    },
                    "a", new String[][]{{"er", ""}, {"est", ""}, {"er", "e"}, {"est", "e"}},
                    "r", new String[][]{});

    private final Path directory;
    private final Map<String, Map<String, List<Integer>>> offsets = new HashMap<>();
    private final Map<String, Map<String, List<String>>> exceptions = new HashMap<>();
    private final Map<String, byte[]> data = new ConcurrentHashMap<>();

    public WordNet(Path directory) throws IOException {
        this.directory = directory;
        for (String pos : PARTS) {
            Map<String, List<String>> partExceptions = new HashMap<>();
            Path exceptionPath = directory.resolve(SUFFIX.get(pos) + ".exc");
            if (Files.exists(exceptionPath)) {
                for (String line : Files.readAllLines(exceptionPath)) {
                    String[] fields = line.strip().split("\\s+");
                    if (fields.length >= 2) {
                        partExceptions.put(fields[0], List.of(fields).subList(1, fields.length));
                    }
                }
            }
            exceptions.put(pos, partExceptions);
            for (String line : Files.readAllLines(directory.resolve("index." + SUFFIX.get(pos)))) {
                loadIndexLine(line);
            }
        }
    }

    private void loadIndexLine(String line) {
        if (line.isEmpty() || line.charAt(0) == ' ') return;
        String[] fields = line.strip().split("\\s+");
        if (fields.length < 6) return;
        try {
            int count = Integer.parseInt(fields[2]);
            int index = 4 + Integer.parseInt(fields[3]);
            Integer.parseInt(fields[index++]);
            Integer.parseInt(fields[index++]);
            List<Integer> values = new ArrayList<>();
            for (int i = 0; i < count && index < fields.length; i++, index++) {
                try {
                    values.add(Integer.parseInt(fields[index]));
                } catch (NumberFormatException ignored) {
                    /* Upstream skips malformed offsets. */
                }
            }
            offsets
                    .computeIfAbsent(fields[0].toLowerCase(Locale.ROOT), ignored -> new HashMap<>())
                    .put(fields[1], List.copyOf(values));
        } catch (NumberFormatException | IndexOutOfBoundsException ignored) {
            // Match the Go index reader's skip-on-malformed-record policy.
        }
    }

    public List<String> morphy(String form, String pos, boolean checkExceptions) {
        form = form.toLowerCase(Locale.ROOT);
        List<String> forms =
                checkExceptions
                        ? exceptions.getOrDefault(pos, Map.of()).getOrDefault(form, List.of())
                        : List.of();
        if (forms.isEmpty()) {
            forms = new ArrayList<>();
            for (String[] rule : RULES.getOrDefault(pos, new String[0][])) {
                if (form.endsWith(rule[0])) {
                    forms.add(form.substring(0, form.length() - rule[0].length()) + rule[1]);
                }
            }
        }
        LinkedHashSet<String> candidates = new LinkedHashSet<>();
        candidates.add(form);
        candidates.addAll(forms);
        candidates.removeIf(candidate -> !offsets.getOrDefault(candidate, Map.of()).containsKey(pos));
        return new ArrayList<>(candidates);
    }

    public List<Synset> synsets(String lemma, String pos) {
        List<Synset> result = new ArrayList<>();
        var seen = new HashSet<String>();
        List<String> parts = pos == null || pos.isEmpty() ? PARTS : List.of(pos);
        for (String part : parts) {
            for (String form : morphy(lemma, part, true)) {
                for (int offset : offsets.getOrDefault(form, Map.of()).getOrDefault(part, List.of())) {
                    if (seen.add(part + "-" + offset)) {
                        try {
                            result.add(synsetAt(part, offset));
                        } catch (IOException | IllegalArgumentException ignored) {
                            // Individual missing/malformed synsets are skipped by the Go reader.
                        }
                    }
                }
            }
        }
        return result;
    }

    private Synset synsetAt(String pos, int offset) throws IOException {
        String filePos = pos.equals("s") ? "a" : pos;
        byte[] bytes = data.get(filePos);
        if (bytes == null) {
            synchronized (data) {
                bytes = data.get(filePos);
                if (bytes == null) {
                    bytes = Files.readAllBytes(directory.resolve("data." + SUFFIX.get(filePos)));
                    data.put(filePos, bytes);
                }
            }
        }
        if (offset < 0 || offset >= bytes.length) throw new IOException("Invalid WordNet offset");
        int end = offset;
        while (end < bytes.length && bytes[end] != '\n') end++;
        if (end == bytes.length) throw new IOException("Truncated WordNet data record");
        String line = new String(bytes, offset, end - offset, StandardCharsets.UTF_8);
        if (line.length() < 8
                || !line.substring(0, 8).strip().equals(String.format(Locale.ROOT, "%08d", offset))) {
            throw new IOException("WordNet offset mismatch");
        }
        Synset parsed = parseDataLine(line);
        String firstLemma = parsed.lemmas().getFirst();
        List<Integer> senses =
                offsets
                        .getOrDefault(firstLemma.toLowerCase(Locale.ROOT), Map.of())
                        .getOrDefault(pos, List.of());
        int senseNumber = Math.max(1, senses.indexOf(offset) + 1);
        String name = String.format(Locale.ROOT, "%s.%s.%02d", firstLemma, parsed.pos(), senseNumber);
        return new Synset(
                name, parsed.pos(), offset, parsed.lemmas(), parsed.definition(), parsed.examples());
    }

    static Synset parseDataLine(String line) {
        String[] parts = line.split("\\|", 2);
        if (parts.length != 2) throw new IllegalArgumentException("WordNet data has no gloss");
        String glossPart = parts[1].strip();
        String gloss = glossPart;
        String definition = "";
        List<String> examples = new ArrayList<>();
        int start;
        while ((start = gloss.indexOf('"')) >= 0) {
            int end = gloss.indexOf('"', start + 1);
            if (end < 0) break;
            if (examples.isEmpty() && start > 0) definition = gloss.substring(0, start).strip();
            examples.add(gloss.substring(start + 1, end));
            gloss = gloss.substring(end + 1);
        }
        if (definition.isEmpty()) {
            StringBuilder remaining = new StringBuilder();
            boolean quoted = false;
            for (char ch : glossPart.toCharArray()) {
                if (ch == '"') quoted = !quoted;
                else if (!quoted) remaining.append(ch);
            }
            definition = remaining.toString().replaceAll("^[; ]+|[; ]+$", "").strip();
        }
        definition = definition.replaceAll("[; ]+$", "");
        String[] fields = parts[0].strip().split("\\s+");
        if (fields.length < 4) throw new IllegalArgumentException("Short WordNet record");
        int offset = Integer.parseInt(fields[0]);
        int count = Integer.parseInt(fields[3], 16);
        List<String> lemmas = new ArrayList<>();
        for (int i = 0, index = 4; i < count && index + 1 < fields.length; i++, index += 2) {
            String lemma = fields[index];
            int marker = lemma.indexOf('(');
            lemmas.add(marker < 0 ? lemma : lemma.substring(0, marker));
        }
        if (lemmas.isEmpty()) throw new IllegalArgumentException("WordNet record has no lemmas");
        return new Synset(
                lemmas.getFirst().toLowerCase(Locale.ROOT) + "." + fields[2] + ".01",
                fields[2],
                offset,
                List.copyOf(lemmas),
                definition,
                List.copyOf(examples));
    }
}
