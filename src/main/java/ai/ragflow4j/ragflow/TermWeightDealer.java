// Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
// http://www.apache.org/licenses/LICENSE-2.0
package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Pattern;

/**
 * 原 internal/service/nlp/term_weight.go，词频、文档频率、NER 与 POS 加权并归一化。
 */
public final class TermWeightDealer {
    private interface LibMath extends Library {
        double log10(double value);
    }

    private static final LibMath LIBM =
            Native.load(Platform.isLinux() ? "m" : Platform.C_LIBRARY_NAME, LibMath.class);
    private static final Pattern NUM_LIKE = Pattern.compile("^[0-9,.]{2,}$");
    private static final Pattern NUM_SPACE = Pattern.compile("^[0-9. -]{2,}$");
    private static final Pattern SHORT_LETTERS = Pattern.compile("^[a-z]{1,2}$");
    private static final Pattern SHORT_ALNUM = Pattern.compile("^[0-9a-z]{1,2}$");
    private static final Pattern LEADING_ALNUM = Pattern.compile("^[0-9a-zA-Z]");
    private static final Pattern ENDS_LETTER = Pattern.compile("[a-zA-Z]$");
    private static final Map<String, Double> ENTITY_WEIGHTS =
            Map.of(
                    "toxic", 2.0, "func", 1.0, "corp", 3.0, "loca", 3.0, "sch", 3.0, "stock", 3.0, "firstnm",
                    1.0);
    private final NativeTokenizer tokenizer;
    private final Set<String> stopWords =
            new HashSet<>(
                    List.of(
                            "请问", "您", "你", "我", "他", "是", "的", "就", "有", "于", "及", "即", "在", "为", "最", "从", "以",
                            "了", "将", "与", "吗", "吧", "中", "#", "什么", "怎么", "哪个", "哪些", "啥", "相关"));
    private final Map<String, String> ne;
    private final Map<String, Integer> df;

    public TermWeightDealer(NativeTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        try (var input = TermWeightDealer.class.getResourceAsStream("/ragflow/ner.json")) {
            ne = new ObjectMapper().readValue(input, new TypeReference<>() {
            });
        } catch (IOException error) {
            throw new IllegalStateException("Cannot load RAGFlow ner.json", error);
        }
        // 当前 Go 仓库没有 term.freq；加载缺失文件时文档频率表为空。
        df = new HashMap<>();
    }

    public TermWeightDealer(NativeTokenizer tokenizer, Path resources) throws IOException {
        this.tokenizer = tokenizer;
        Path nerPath = resources.resolve("ner.json");
        ne =
                Files.exists(nerPath)
                        ? new ObjectMapper().readValue(Files.readString(nerPath), new TypeReference<>() {
                })
                        : new HashMap<>();
        df = loadDict(resources.resolve("term.freq"));
    }

    public Map<String, String> getNE() {
        return ne;
    }

    public Map<String, Integer> getDF() {
        return df;
    }

    public Set<String> getStopWords() {
        return stopWords;
    }

    public String ner(String term) {
        return ne.getOrDefault(term, "");
    }

    static Map<String, Integer> loadDict(Path path) throws IOException {
        Map<String, Integer> result = new HashMap<>();
        if (!Files.exists(path)) return result;
        for (String line : Files.readString(path).split("\n")) {
            line = line.strip();
            if (line.isEmpty()) continue;
            String[] parts = line.split("\t", -1);
            if (parts.length == 1) result.put(parts[0], 0);
            else {
                try {
                    result.put(parts[0], Integer.parseInt(parts[1]));
                } catch (NumberFormatException ignored) {
                }
            }
        }
        return result;
    }

    public List<String> pretoken(String text, boolean num, boolean removeStopWords, String language) {
        String tokenized;
        try {
            tokenized = tokenizer.tokenize(text, language);
        } catch (RuntimeException error) {
            tokenized = text;
        }
        List<String> result = new ArrayList<>();
        // 原 pretokenPunctPattern 无法编译，其标点替换分支始终不执行。
        for (String term : Reranker.split(tokenized)) {
            if (removeStopWords && stopWords.contains(term)) continue;
            if (!num && term.matches("^[0-9]$")) continue;
            if (!term.equals("#")) result.add(term);
        }
        return result;
    }

    public List<String> tokenMerge(List<String> tokens) {
        List<String> result = new ArrayList<>();
        int i = 0;
        while (i < tokens.size()) {
            if (i == 0
                    && tokens.size() > 1
                    && oneTerm(tokens.get(i))
                    && runeCount(tokens.get(i + 1)) > 1
                    && !LEADING_ALNUM.matcher(tokens.get(i + 1)).find()) {
                result.add(tokens.get(0) + " " + tokens.get(1));
                i = 2;
                continue;
            }
            int j = i;
            while (j < tokens.size()
                    && !tokens.get(j).isEmpty()
                    && !stopWords.contains(tokens.get(j))
                    && oneTerm(tokens.get(j))) j++;
            if (j - i > 1) {
                if (j - i < 5) result.add(String.join(" ", tokens.subList(i, j)));
                else {
                    for (int k = i; k < j; k += 2) {
                        if (k + 1 < j) result.add(tokens.get(k) + " " + tokens.get(k + 1));
                        else result.add(tokens.get(k));
                    }
                }
                i = j;
            } else {
                if (!tokens.get(i).isEmpty()) result.add(tokens.get(i));
                i++;
            }
        }
        return result;
    }

    private static boolean oneTerm(String term) {
        return runeCount(term) == 1 || SHORT_ALNUM.matcher(term).matches();
    }

    static int runeCount(String term) {
        return term.codePointCount(0, term.length());
    }

    public List<String> split(String text) {
        if (text.isEmpty()) return new ArrayList<>(List.of(""));
        List<String> tokens = new ArrayList<>();
        for (String term : text.replaceAll("[ \\t]+", " ").strip().split(" ", -1)) {
            term = term.strip();
            if (term.isEmpty()) continue;
            if (!tokens.isEmpty()) {
                String previous = tokens.getLast();
                if (ENDS_LETTER.matcher(previous).find()
                        && ENDS_LETTER.matcher(term).find()
                        && !ner(term).equals("func")
                        && !ner(previous).equals("func")) {
                    tokens.set(tokens.size() - 1, previous + " " + term);
                    continue;
                }
            }
            tokens.add(term);
        }
        return tokens;
    }

    static Double alphabeticOOVFrequency(String term) {
        int letters = 0;
        for (int codePoint : term.codePoints().toArray()) {
            Character.UnicodeScript script = Character.UnicodeScript.of(codePoint);
            if (Character.isLetter(codePoint)
                    && (script == Character.UnicodeScript.LATIN
                    || script == Character.UnicodeScript.GREEK
                    || script == Character.UnicodeScript.CYRILLIC)) {
                letters++;
            } else if (codePoint != ' ' && codePoint != '.' && codePoint != '-') return null;
        }
        if (letters == 0) return null;
        double exponent = Math.max(0, letters - 3) / 2.0;
        return Math.max(10, Math.floor(300 / Math.pow(2, exponent) + 0.5));
    }

    private double nerWeight(String term) {
        if (NUM_LIKE.matcher(term).matches()) return 2;
        if (SHORT_LETTERS.matcher(term).matches()) return 0.01;
        return ENTITY_WEIGHTS.getOrDefault(ner(term), 1.0);
    }

    private double postagWeight(String tag) {
        if (tag.equals("r") || tag.equals("c") || tag.equals("d")) return 0.3;
        if (tag.equals("ns") || tag.equals("nt")) return 3;
        if (tag.equals("n") || tag.matches("^[0-9-]+.*")) return 2;
        return 1;
    }

    private List<String> fineTokens(String term, String language) {
        String fine;
        try {
            fine = tokenizer.fineGrainedTokenize(term, language);
        } catch (RuntimeException error) {
            fine = "";
        }
        return Reranker.split(fine).stream().filter(token -> runeCount(token) > 1).toList();
    }

    private double frequency(String term, String language, Map<String, Integer> frequencies) {
        if (NUM_SPACE.matcher(term).matches()) return 3;
        int frequency = frequencies.computeIfAbsent(term, tokenizer::frequency);
        if (frequency == 0) {
            Double alphabetic = alphabeticOOVFrequency(term);
            if (alphabetic != null) return alphabetic;
        }
        if (frequency == 0 && runeCount(term) >= 4) {
            List<String> fine = fineTokens(term, language);
            if (fine.size() > 1) {
                double minimum = Double.POSITIVE_INFINITY;
                for (String token : fine)
                    minimum = Math.min(minimum, frequency(token, language, frequencies));
                return minimum / 6;
            }
            return 10;
        }
        return Math.max(frequency, 10);
    }

    private double documentFrequency(String term, String language) {
        if (NUM_SPACE.matcher(term).matches()) return 5;
        if (df.containsKey(term)) return df.get(term) + 3.0;
        Double alphabetic = alphabeticOOVFrequency(term);
        if (alphabetic != null) return alphabetic;
        if (runeCount(term) >= 4) {
            List<String> fine = fineTokens(term, language);
            if (fine.size() > 1) {
                double minimum = Double.POSITIVE_INFINITY;
                for (String token : fine) minimum = Math.min(minimum, documentFrequency(token, language));
                return Math.max(3, minimum / 6);
            }
        }
        return 3;
    }

    /**
     * 为词项估计相对重要性：结合词频、文档频率、实体类别（NER）与词性（POS），再归一化。
     *
     * <p>词频来自原生词典，文档频率来自资源表；这里不会实时统计当前 ES 数据集的词频。
     */
    public List<Reranker.TermWeight> weights(
            List<String> tokens, boolean preprocess, String language) {
        List<String> terms = new ArrayList<>();
        if (tokens == null) return new ArrayList<>();
        if (preprocess) {
            for (String token : tokens) terms.addAll(tokenMerge(pretoken(token, true, true, language)));
        } else terms.addAll(tokens);
        Map<String, Integer> frequencies = new HashMap<>();
        Map<String, String> tags = new HashMap<>();
        double[] values = new double[terms.size()];
        for (int i = 0; i < terms.size(); i++) {
            String term = terms.get(i);
            double idf1 = idf(frequency(term, language, frequencies), 10_000_000);
            double idf2 = idf(documentFrequency(term, language), 1_000_000_000);
            double nerPos = nerWeight(term) * postagWeight(tags.computeIfAbsent(term, tokenizer::tag));
            // 两种 IDF（逆频率）加权后乘实体/词性系数；具体常数沿用原算法。
            values[i] = (0.3 * idf1 + 0.7 * idf2) * nerPos;
        }
        double total = Reranker.pairwiseSum(values.clone());
        List<Reranker.TermWeight> result = new ArrayList<>();
        for (int i = 0; i < terms.size(); i++)
            result.add(new Reranker.TermWeight(terms.get(i), total > 0 ? values[i] / total : values[i]));
        return result;
    }

    private static double idf(double frequency, double total) {
        // 原 common.PyLog10 走系统 libm，避免 Java 实现导致的末位差异。
        return LIBM.log10(10 + (total - frequency + 0.5) / (frequency + 0.5));
    }
}
