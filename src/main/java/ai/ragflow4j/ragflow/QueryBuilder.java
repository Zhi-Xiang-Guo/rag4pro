// Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
// http://www.apache.org/licenses/LICENSE-2.0
package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;
import java.util.regex.Pattern;

/**
 * 将自然语言问题编译为带字段权重和词权重的词法检索表达式。
 *
 * <p>对应 internal/service/nlp/query_builder.go 的 Question 路径；这里只构造表达式，不发 ES 请求。 同一类还为 Reranker
 * 提供查询词与词权重，保证召回和二阶段评分使用一致的词法处理。
 */
public final class QueryBuilder implements Reranker.TokenSource {
    private static final Map<String, Pattern> PATTERNS = loadPatterns();
    private static final Map<Integer, Integer> TRADITIONAL_TO_SIMPLIFIED = loadCharacterMapping();
    // ^ 后是字段 boost：同一个词出现在标题、关键词、正文中，对召回得分的贡献不同。
    private List<String> queryFields =
            List.of(
                    "title_tks^10",
                    "title_sm_tks^5",
                    "important_kwd^30",
                    "important_tks^20",
                    "question_tks^20",
                    "content_ltks^2",
                    "content_sm_ltks");
    private final NativeTokenizer tokenizer;
    private final TermWeightDealer termWeight;
    private final Synonym synonym;

    /**
     * matchingText 是带 boost/短语的 query_string 内容，extraOptions 携带最低匹配比例等选项。
     */
    public record MatchTextExpr(
            List<String> fields, String matchingText, int topN, Map<String, Object> extraOptions) {
    }

    /**
     * expression 交给 ES；keywords 留给高亮等后续处理，不是 embedding 向量。
     */
    public record QuestionResult(MatchTextExpr expression, List<String> keywords) {
    }

    private record WeightedToken(String term, double weight) {
    }

    public QueryBuilder(NativeTokenizer tokenizer) {
        this(tokenizer, new TermWeightDealer(tokenizer), new Synonym());
    }

    public QueryBuilder(NativeTokenizer tokenizer, TermWeightDealer termWeight, Synonym synonym) {
        this.tokenizer = tokenizer;
        this.termWeight = termWeight;
        this.synonym = synonym;
    }

    public void setQueryFields(List<String> fields) {
        queryFields = List.copyOf(fields);
    }

    public TermWeightDealer termWeight() {
        return termWeight;
    }

    @Override
    public List<String> question(String query, String language) {
        return question(query, "qa", 0.6, language).keywords();
    }

    @Override
    public List<Reranker.TermWeight> weights(List<String> tokens, String language) {
        return termWeight.weights(tokens, false, language);
    }

    private static Map<String, Pattern> loadPatterns() {
        try (var input = QueryBuilder.class.getResourceAsStream("/ragflow/query-patterns.json")) {
            Map<String, String> source = new ObjectMapper().readValue(input, new TypeReference<>() {
            });
            Map<String, Pattern> result = new HashMap<>();
            source.forEach((key, value) -> result.put(key, Pattern.compile(value)));
            return result;
        } catch (IOException error) {
            throw new ExceptionInInitializerError(error);
        }
    }

    /**
     * gojianfan v0.0.0-20210926212422-2f175ac615de 的原始字符表，逐 Unicode 字符映射。
     */
    private static Map<Integer, Integer> loadCharacterMapping() {
        try (var input = QueryBuilder.class.getResourceAsStream("/ragflow/jianfan.json")) {
            Map<String, String> source = new ObjectMapper().readValue(input, new TypeReference<>() {
            });
            int[] traditional = source.get("ChT").codePoints().toArray();
            int[] simplified = source.get("ChS").codePoints().toArray();
            if (traditional.length != simplified.length)
                throw new IllegalStateException("jianfan character table mismatch");
            Map<Integer, Integer> result = new HashMap<>();
            for (int i = 0; i < traditional.length; i++) result.put(traditional[i], simplified[i]);
            return result;
        } catch (IOException error) {
            throw new ExceptionInInitializerError(error);
        }
    }

    private static String replace(String pattern, String text, String replacement) {
        return PATTERNS.get(pattern).matcher(text).replaceAll(replacement);
    }

    private static boolean matches(String pattern, String text) {
        return PATTERNS.get(pattern).matcher(text).find();
    }

    public boolean isChinese(String line) {
        List<String> fields = Reranker.split(line);
        if (fields.size() <= 3) return true;
        long nonAlpha = fields.stream().filter(field -> !matches("reEngAlpha", field)).count();
        return (double) nonAlpha / fields.size() >= 0.7;
    }

    public String subSpecialChar(String line) {
        return replace("reSubSpecialChar", line.replace("'", ""), "\\\\$1").strip();
    }

    public String rmWWW(String text) {
        String original = text;
        text = replace("reStopWordsZH", text, "");
        text = replace("reStopWordsEN1", text, " ");
        text = replace("reStopWordsEN2", text, " ");
        return text.isEmpty() ? original : text;
    }

    public String addSpaceBetweenEngZh(String text) {
        for (String pattern : List.of("reEngZhNum", "reEngZh", "reZhEngNum", "reZhEng"))
            text = replace(pattern, text, "$1 $2");
        return text;
    }

    public String strFullWidth2HalfWidth(String text) {
        StringBuilder result = new StringBuilder();
        text.codePoints()
                .forEach(
                        codePoint -> {
                            int converted = codePoint == 0x3000 ? 0x20 : codePoint - 0xFEE0;
                            result.appendCodePoint(converted < 0x20 || converted > 0x7E ? codePoint : converted);
                        });
        return result.toString();
    }

    public String traditional2Simplified(String text) {
        StringBuilder result = new StringBuilder();
        text.codePoints()
                .forEach(
                        codePoint ->
                                result.appendCodePoint(
                                        TRADITIONAL_TO_SIMPLIFIED.getOrDefault(codePoint, codePoint)));
        return result.toString();
    }

    public boolean needFineGrainedTokenize(String term) {
        return TermWeightDealer.runeCount(term) >= 3 && !matches("reSimpleToken", term);
    }

    private String tokenizeOrOriginal(String text, String language) {
        try {
            return tokenizer.tokenize(text, language);
        } catch (RuntimeException error) {
            return text;
        }
    }

    private String tokenizeOrEmpty(String text, String language) {
        try {
            return tokenizer.tokenize(text, language);
        } catch (RuntimeException error) {
            return "";
        }
    }

    private String fineGrainedOrEmpty(String text, String language) {
        try {
            return tokenizer.fineGrainedTokenize(text, language);
        } catch (RuntimeException error) {
            return "";
        }
    }

    List<String> fineGrainedKeywords(String text, String language) {
        return Reranker.split(fineGrainedOrEmpty(text, language));
    }

    /**
     * 原 Question，tbl 参数保留原签名，当前 Go 实现未使用。
     */
    public QuestionResult question(String text, String tbl, double minMatch, String language) {
        String originalQuery = text;
        StringBuilder lowercase = new StringBuilder();
        addSpaceBetweenEngZh(text)
                .codePoints()
                .forEach(codePoint -> lowercase.appendCodePoint(Character.toLowerCase(codePoint)));
        // 先统一大小写、全半角和繁简，再清理标点与停用词，使不同写法更容易匹配。
        String normalized = traditional2Simplified(strFullWidth2HalfWidth(lowercase.toString()));
        String cleaned = rmWWW(replace("rePunct", normalized, " "));
        // 这是原版基于文本特征的中英文路径判定，不是独立的语言识别模型。
        if (!isChinese(cleaned)) return englishQuestion(cleaned, originalQuery, language);
        return chineseQuestion(cleaned, originalQuery, minMatch, language);
    }

    private QuestionResult englishQuestion(String text, String originalQuery, String language) {
        String finalText = rmWWW(text);
        List<String> tokens = Reranker.split(tokenizeOrOriginal(finalText, language));
        List<String> keywords = new ArrayList<>(tokens);
        List<WeightedToken> weighted = new ArrayList<>();
        for (Reranker.TermWeight term : termWeight.weights(tokens, false, language)) {
            String cleaned = replace("reCleanQuote", term.term(), "");
            cleaned = replace("reSingleChar", cleaned, "");
            cleaned = replace("reLeadSign", cleaned, "").strip();
            if (!cleaned.isEmpty()) weighted.add(new WeightedToken(cleaned, term.weight()));
        }
        if (weighted.size() > 256) weighted = weighted.subList(0, 256);
        List<String> synonyms = new ArrayList<>();
        for (WeightedToken term : weighted) {
            List<String> parts = new ArrayList<>();
            for (String value : synonym.lookup(term.term(), 8)) {
                // 原 cleanSynonym 使用默认 English 分词器。
                String cleaned = tokenizeOrOriginal(value, "").replace("'", "").strip();
                if (!cleaned.isEmpty()) {
                    parts.add("\"" + cleaned + "\"^" + boost(term.weight() / 4));
                    keywords.add(cleaned);
                }
            }
            synonyms.add(String.join(" ", parts));
        }
        List<String> query = new ArrayList<>();
        for (int i = 0; i < weighted.size(); i++) {
            WeightedToken term = weighted.get(i);
            if (matches("reQuerySpecial", term.term())) continue;
            query.add("(" + term.term() + "^" + boost(term.weight()) + " " + synonyms.get(i) + ")");
        }
        for (int i = 1; i < weighted.size(); i++) {
            WeightedToken left = weighted.get(i - 1), right = weighted.get(i);
            if (left.term().strip().isEmpty() || right.term().strip().isEmpty()) continue;
            double maximum = left.weight();
            if (right.weight() > maximum) maximum = right.weight();
            query.add(
                    "\"" + left.term().strip() + " " + right.term().strip() + "\"^" + boost(maximum * 2));
        }
        if (query.isEmpty()) query.add(finalText);
        return new QuestionResult(
                new MatchTextExpr(
                        queryFields, String.join(" ", query), 100, Map.of("original_query", originalQuery)),
                keywords);
    }

    private QuestionResult chineseQuestion(
            String text, String originalQuery, double minMatch, String language) {
        List<String> queries = new ArrayList<>();
        List<String> keywords = new ArrayList<>();
        List<String> segments = termWeight.split(rmWWW(text));
        if (segments.size() > 256) segments = segments.subList(0, 256);
        for (String segment : segments) {
            if (segment.isEmpty()) continue;
            keywords.add(segment);
            List<Reranker.TermWeight> weighted = termWeight.weights(List.of(segment), true, language);
            List<String> segmentSynonyms = synonym.lookup(segment, 8);
            if (!segmentSynonyms.isEmpty() && keywords.size() < 32) keywords.addAll(segmentSynonyms);
            weighted.sort((left, right) -> Double.compare(right.weight(), left.weight()));
            List<WeightedToken> terms = new ArrayList<>();
            for (Reranker.TermWeight item : weighted) {
                String term = item.term();
                List<String> fine = new ArrayList<>();
                if (needFineGrainedTokenize(term)) {
                    for (String token : Reranker.split(fineGrainedOrEmpty(term, language))) {
                        String cleaned = subSpecialChar(replace("reSpecialChar", token, ""));
                        if (TermWeightDealer.runeCount(cleaned) > 1) fine.add(cleaned);
                    }
                }
                if (keywords.size() < 32) {
                    String cleaned = replace("reCleanTerm", term, "");
                    if (!cleaned.isEmpty()) keywords.add(cleaned);
                    keywords.addAll(fine);
                }
                List<String> tokenSynonyms =
                        synonym.lookup(term, 8).stream().map(this::subSpecialChar).toList();
                if (keywords.size() < 32) {
                    for (String value : tokenSynonyms) if (!value.isEmpty()) keywords.add(value);
                }
                List<String> fineSynonyms = new ArrayList<>();
                for (String value : tokenSynonyms) {
                    if (value.isEmpty()) continue;
                    String tokenized = fineGrainedOrEmpty(value, language);
                    if (!tokenized.isEmpty()) fineSynonyms.add(quoteSpaces(tokenized));
                }
                // 原分支在构建当前项之前检查32，已加入的关键词仍然保留。
                if (keywords.size() >= 32) break;
                term = subSpecialChar(term);
                if (term.isEmpty()) continue;
                term = quoteSpaces(term);
                if (!fineSynonyms.isEmpty())
                    term = "(" + term + " OR (" + String.join(" ", fineSynonyms) + ")^0.2)";
                if (!fine.isEmpty()) {
                    String joined = String.join(" ", fine);
                    term += " OR \"" + joined + "\" OR (\"" + joined + "\"~2)^0.5";
                }
                terms.add(new WeightedToken(term, item.weight()));
            }
            List<String> parts = new ArrayList<>();
            for (WeightedToken term : terms) parts.add("(" + term.term() + ")^" + goFloat(term.weight()));
            String expression = String.join(" ", parts);
            if (weighted.size() > 1) {
                String tokenized = tokenizeOrEmpty(segment, language);
                if (!tokenized.isEmpty()) expression += " (\"" + tokenized + "\"~2)^1.5";
            }
            if (!segmentSynonyms.isEmpty() && !expression.isEmpty()) {
                List<String> synonymParts = new ArrayList<>();
                for (String value : segmentSynonyms) {
                    value = subSpecialChar(value);
                    if (!value.isEmpty()) {
                        String tokenized = tokenizeOrEmpty(value, language);
                        if (!tokenized.isEmpty()) synonymParts.add("\"" + tokenized + "\"");
                    }
                }
                if (!synonymParts.isEmpty())
                    expression = "(" + expression + ")^5 OR (" + String.join(" OR ", synonymParts) + ")^0.7";
            }
            if (!expression.isEmpty()) queries.add(expression);
        }
        if (queries.isEmpty()) return new QuestionResult(null, keywords);
        List<String> wrapped =
                queries.stream().filter(query -> !query.isEmpty()).map(query -> "(" + query + ")").toList();
        String query = String.join(" OR ", wrapped);
        if (query.isEmpty()) query = text;
        return new QuestionResult(
                new MatchTextExpr(
                        queryFields,
                        query,
                        100,
                        Map.of("minimum_should_match", minMatch, "original_query", originalQuery)),
                keywords);
    }

    private static String quoteSpaces(String term) {
        return term.contains(" ") ? "\"" + term + "\"" : term;
    }

    // Go fmt 的十进制舍入采用 ties-to-even，直接保留输入 double 的精确值。
    static String boost(double value) {
        if (!Double.isFinite(value)) return goFloat(value);
        return new BigDecimal(value).setScale(4, RoundingMode.HALF_EVEN).toPlainString();
    }

    // Go fmt %v 的浮点表示不保留整数的小数部分，科学计数使用小写e及带符号指数。
    static String goFloat(double value) {
        if (!Double.isFinite(value)) return Double.toString(value).replace("Infinity", "Inf");
        BigDecimal decimal = BigDecimal.valueOf(value).stripTrailingZeros();
        int power = decimal.precision() - decimal.scale() - 1;
        if (power < -4 || power >= 6) {
            String mantissa = decimal.movePointLeft(power).toPlainString();
            return mantissa + "e" + String.format(Locale.ROOT, "%+03d", power);
        }
        return decimal.toPlainString();
    }

    /**
     * 原 Paragraph 当前仅使用关键词，contentTks 不参与表达式。
     */
    public MatchTextExpr paragraph(String contentTks, List<String> keywords, int keywordsTopN) {
        List<String> terms = new ArrayList<>();
        for (String term : keywords) if (!term.strip().isEmpty()) terms.add("\"" + term.strip() + "\"");
        if (keywordsTopN > 0 && terms.size() > keywordsTopN) terms = terms.subList(0, keywordsTopN);
        return new MatchTextExpr(queryFields, String.join(" ", terms), 100, Map.of());
    }
}
