// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.
// http://www.apache.org/licenses/LICENSE-2.0
package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.databind.*;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.util.*;
import java.util.regex.Pattern;

/**
 * 在 chunk 召回前，按文档 metadata 缩小文档 ID 范围。
 *
 * <p>对应 service/metadata_filter.go 和 common/metadata_utils.go：manual 使用给定条件， auto 由模型生成条件，semi_auto
 * 先限制向模型展示的字段。最终仍由本地条件比较逻辑执行筛选。
 */
public final class MetadataFilter {
    public static final String NO_MATCH_DOC_ID = "-999";
    private static final ObjectMapper JSON =
            new ObjectMapper().enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
    private static final Map<String, String> OPERATORS =
            Map.of("is", "=", "not is", "≠", ">=", "≥", "<=", "≤", "!=", "≠", "==", "=");
    private static final Pattern CONSTRAINT_BLOCK =
            Pattern.compile(
                    "\\{%\\s*if\\s+constraints\\s*%\\}(.+?)\\{%\\s*endif\\s*%\\}", Pattern.DOTALL);
    private static final Pattern DATE = Pattern.compile("[0-9]{4}-[0-9]{2}-[0-9]{2}");
    private static final String DECIMAL_DIGITS = "[0-9](?:_?[0-9])*";
    private static final String HEX_DIGITS = "[0-9a-fA-F](?:_?[0-9a-fA-F])*";
    private static final Pattern FLOAT =
            Pattern.compile(
                    "[+-]?(?:(?:"
                            + DECIMAL_DIGITS
                            + "(?:\\.(?:"
                            + DECIMAL_DIGITS
                            + ")?)?|\\."
                            + DECIMAL_DIGITS
                            + ")(?:[eE][+-]?"
                            + DECIMAL_DIGITS
                            + ")?|0[xX]_?(?:"
                            + HEX_DIGITS
                            + "(?:\\.(?:"
                            + HEX_DIGITS
                            + ")?)?|\\."
                            + HEX_DIGITS
                            + ")[pP][+-]?"
                            + DECIMAL_DIGITS
                            + ")");
    private static final String PROMPT =
            """
                    You are a metadata filtering condition generator. Analyze the user's question and available document metadata to output a JSON array of filter objects. Follow these rules:
                    
                    1. **Metadata Structure**:
                       - Metadata is provided as JSON where keys are attribute names (e.g., "color"), and values are objects mapping attribute values to document IDs.
                       - Example:
                         {
                           "color": {"red": ["doc1"], "blue": ["doc2"]},
                           "listing_date": {"2025-07-11": ["doc1"], "2025-08-01": ["doc2"]}
                         }
                    
                    2. **Output Requirements**:
                       - Always output a JSON dictionary with only 2 keys: 'conditions'(filter objects) and 'logic' between the conditions ('and' or 'or').
                       - Each filter object in conditions must have:
                            "key": (metadata attribute name),
                            "value": (string value to compare),
                            "op": (operator from allowed list)
                       - Logic between all the conditions: 'and'(Intersection of results for each condition) / 'or' (union of results for all conditions)
                    
                    
                    3. **Operator Guide**:
                       - Use these operators only: ["contains", "not contains","in", "not in", "start with", "end with", "empty", "not empty", "=", "≠", ">", "<", "≥", "≤"]
                       - Date ranges: Break into two conditions (≥ start_date AND < next_month_start)
                       - Negations: Always use "≠" for exclusion terms ("not", "except", "exclude", "≠")
                       - Implicit logic: Derive unstated filters (e.g., "July" → [≥ YYYY-07-01, < YYYY-08-01])
                    
                    4. **Operator Constraints**:
                       - If `constraints` are provided, you MUST use the specified operator for the corresponding key.
                       - Example Constraints: `{"price": ">", "author": "="}`
                       - If a key is not in `constraints`, choose the most appropriate operator.
                    
                    5. **Processing Steps**:
                       a) Identify ALL filterable attributes in the query (both explicit and implicit)
                       b) For dates:
                            - Infer missing year from current date if needed
                            - Always format dates as "YYYY-MM-DD"
                            - Convert ranges: [≥ start, < end]
                       c) For values: Match EXACTLY to metadata's value keys
                       d) Skip conditions if:
                            - Attribute doesn't exist in metadata
                            - Value has no match in metadata
                    
                    6. **Example A**:
                       - User query: "上市日期七月份的有哪些新品，不要蓝色的，只看鞋子和帽子"
                       - Metadata: { "color": {...}, "listing_date": {...} }
                       - Output:
                       {
                            "logic": "and",
                            "conditions": [
                              {"key": "listing_date", "value": "2025-07-01", "op": "≥"},
                              {"key": "listing_date", "value": "2025-08-01", "op": "<"},
                              {"key": "color", "value": "blue", "op": "≠"},
                              {"key": "category", "value": "shoes, hat", "op": "in"}
                            ]
                       }
                    
                    7. **Example B**:
                       - User query: "It must be from China or India. Otherwise, it must not be blue or red."
                       - Metadata: { "color": {...}, "country": {...} }
                       -
                       - Output:
                       {
                            "logic": "or",
                            "conditions": [
                              {"key": "color", "value": "blue, red", "op": "not in"},
                              {"key": "country", "value": "china, india", "op": "in"},
                            ]
                       }
                    
                    8. **Final Output**:
                       - ONLY output valid JSON dictionary
                       - NO additional text/explanations
                       - Json schema is as following:
                    ```json
                    {
                      "type": "object",
                      "properties": {
                        "logic": {
                          "type": "string",
                          "description": "Logic relationship between all the conditions, the default is 'and'.",
                          "enum": [
                            "and",
                            "or"
                          ]
                        },
                        "conditions": {
                          "type": "array",
                          "items": {
                            "type": "object",
                            "properties": {
                              "key": {
                                "type": "string",
                                "description": "Metadata attribute name."
                              },
                              "value": {
                                "type": "string",
                                "description": "Value to compare."
                              },
                              "op": {
                                "type": "string",
                                "description": "Operator from allowed list.",
                                "enum": [
                                  "contains",
                                  "not contains",
                                  "in",
                                  "not in",
                                  "start with",
                                  "end with",
                                  "empty",
                                  "not empty",
                                  "=",
                                  "≠",
                                  ">",
                                  "<",
                                  "≥",
                                  "≤"
                                ]
                              }
                            },
                            "required": [
                              "key",
                              "value",
                              "op"
                            ],
                            "additionalProperties": false
                          }
                        }
                      },
                      "required": [
                        "conditions"
                      ],
                      "additionalProperties": false
                    }
                    ```
                    
                    **Current Task**:
                    - Today's date: {{ current_date }}
                    - Available metadata keys: {{ metadata_keys }}
                    - User query: "{{ user_question }}"
                    {% if constraints %}
                    - Operator constraints: {{ constraints }}
                    {% endif %}
                    """;

    public record Condition(String key, Object value, String op) {
    }

    public record Generated(List<Condition> conditions, String logic) {
    }

    private MetadataFilter() {
    }

    /**
     * 原字段名优先，metadataFilter 为本地 HTTP 拼写。
     */
    public static Map<String, Object> configFromRequest(Map<String, Object> request) {
        Object value =
                request.containsKey("meta_data_filter")
                        ? request.get("meta_data_filter")
                        : request.get("metadataFilter");
        if (value == null) return null;
        if (!(value instanceof Map<?, ?> map))
            throw new IllegalArgumentException("meta_data_filter 必须为对象");
        Map<String, Object> result = new LinkedHashMap<>();
        map.forEach((key, entry) -> result.put(key.toString(), entry));
        return result;
    }

    /**
     * 原 chatMetadataDocIDs 的调用契约，自动过滤没有有效缩窄时保留原文档范围。
     */
    public static List<String> filterDocumentIds(
            Map<String, Object> config,
            List<Map<String, Object>> catalogRows,
            String question,
            ChatPipeline.ChatModel model,
            List<String> baseDocIds) {
        if (config == null || config.isEmpty() || question.isEmpty()) return baseDocIds;
        List<String> filtered =
                applyMetaDataFilter(config, flatten(catalogRows), question, model, baseDocIds);
        return filtered == null ? baseDocIds : filtered;
    }

    /**
     * 原 ApplyMetaDataFilter。null 为未缩窄，手工空命中为 -999，其他结果已与原文档范围求交。
     */
    public static List<String> applyMetaDataFilter(
            Map<String, Object> config,
            Map<String, Map<String, List<String>>> metadata,
            String question,
            ChatPipeline.ChatModel model,
            List<String> baseDocIds) {
        if (config == null) return baseDocIds;
        String method = string(config.get("method"));
        if (method.equals("manual")) {
            List<?> manual = config.get("manual") instanceof List<?> list ? list : List.of();
            if (manual.isEmpty()) return baseDocIds;
            List<Condition> conditions = parseConditions(manual);
            String logic = config.get("logic") instanceof String value ? value : "and";
            List<String> selected =
                    constrainDocIds(baseDocIds, applyMetaFilter(metadata, conditions, logic));
            // 空 docIds 在 ES 表示不限定文档；手工条件零命中必须用哨兵表达，避免误查全库。
            return selected.isEmpty() ? List.of(NO_MATCH_DOC_ID) : selected;
        }
        Map<String, Map<String, List<String>>> promptMetadata = metadata;
        Map<String, String> constraints = null;
        if (method.equals("semi_auto")) {
            promptMetadata = new LinkedHashMap<>();
            constraints = new LinkedHashMap<>();
            List<?> selected = config.get("semi_auto") instanceof List<?> list ? list : List.of();
            for (Object item : selected) {
                String key = "";
                if (item instanceof String text) key = text;
                else if (item instanceof Map<?, ?> map && map.get("key") instanceof String text) {
                    key = text;
                    if (map.get("op") instanceof String operator) constraints.put(key, operator);
                }
                if (metadata.containsKey(key)) promptMetadata.put(key, metadata.get(key));
            }
            if (promptMetadata.isEmpty()) return baseDocIds;
        } else if (!method.equals("auto")) return baseDocIds;
        try {
            Generated generated =
                    generate(promptMetadata, question, constraints, model, LocalDate.now().toString());
            // 半自动只限制给模型看的字段；原版执行生成条件时仍查完整 metadata。
            List<String> selected =
                    constrainDocIds(
                            baseDocIds, applyMetaFilter(metadata, generated.conditions(), generated.logic()));
            return selected.isEmpty() ? null : selected;
        } catch (Exception error) {
            System.getLogger(MetadataFilter.class.getName())
                    .log(
                            System.Logger.Level.WARNING,
                            "Metadata filter generation failed; preserving document scope",
                            error);
            return baseDocIds;
        }
    }

    /**
     * 原 GetFlattedMetaByKBs，加上原写入阶段 SplitCombinedMetadataValues 的列表规范化。
     */
    public static Map<String, Map<String, List<String>>> flatten(
            List<Map<String, Object>> catalogRows) {
        Map<String, Map<String, List<String>>> result = new LinkedHashMap<>();
        for (Map<String, Object> row : catalogRows) {
            String documentId = string(row.get("internalDocId"));
            if (documentId.isEmpty() || !(row.get("metadata") instanceof Map<?, ?> metadata)) continue;
            for (Map.Entry<?, ?> field : metadata.entrySet()) {
                if (!(field.getKey() instanceof String key) || field.getValue() == null) continue;
                Map<String, List<String>> values =
                        result.computeIfAbsent(key, ignored -> new LinkedHashMap<>());
                List<?> entries =
                        field.getValue() instanceof List<?> list
                                ? normalizeCombined(list)
                                : List.of(field.getValue());
                for (Object entry : entries) {
                    if (!(entry instanceof String) && !(entry instanceof Number)) continue;
                    String text = toStringValue(entry);
                    if (entry instanceof String && text.isEmpty()) continue;
                    List<String> documents = values.computeIfAbsent(text, ignored -> new ArrayList<>());
                    if (!documents.contains(documentId)) documents.add(documentId);
                }
            }
        }
        return result;
    }

    private static List<?> normalizeCombined(List<?> items) {
        if (!items.stream().allMatch(String.class::isInstance)) return items;
        Set<String> result = new LinkedHashSet<>();
        for (Object item : items) {
            String text = (String) item;
            if (text.isEmpty()) continue;
            boolean found = false;
            for (String part : text.strip().split("[、,，;；|]+", -1)) {
                part = part.strip();
                if (!part.isEmpty()) {
                    result.add(part);
                    found = true;
                }
            }
            if (!found) result.add(text);
        }
        return new ArrayList<>(result);
    }

    public static List<String> applyMetaFilter(
            Map<String, Map<String, List<String>>> metadata, List<Condition> conditions, String logic) {
        if (conditions == null || conditions.isEmpty()) return List.of();
        boolean intersection = logic == null || logic.isEmpty() || logic.equals("and");
        Set<String> result = null;
        for (Condition condition : conditions) {
            Map<String, List<String>> values = metadata.get(condition.key());
            if (values == null) {
                if (intersection) return List.of();
                continue;
            }
            String operator = OPERATORS.getOrDefault(condition.op(), condition.op());
            Object value = condition.value();
            boolean setOperator = operator.equals("in") || operator.equals("not in");
            Set<String> membership = null;
            if (setOperator) {
                value = setValue(value);
                if (value instanceof List<?> list) {
                    membership = new HashSet<>();
                    for (Object item : list)
                        membership.add(
                                operator.equals("not in") ? lower(toStringValue(item)) : toStringValue(item));
                }
            }
            Set<String> matched = new LinkedHashSet<>();
            for (Map.Entry<String, List<String>> entry : values.entrySet()) {
                boolean match;
                if (setOperator)
                    match =
                            membership != null
                                    && (operator.equals("in")
                                    ? membership.contains(entry.getKey())
                                    : !membership.contains(lower(entry.getKey())));
                else match = matches(entry.getKey(), operator, value);
                if (match) matched.addAll(entry.getValue());
            }
            if (result == null) result = matched;
            else if (intersection) result.retainAll(matched);
            else result.addAll(matched);
            if (intersection && result.isEmpty()) return List.of();
        }
        return result == null ? List.of() : new ArrayList<>(result);
    }

    private static Object setValue(Object value) {
        if (value instanceof List<?> list)
            return list.stream().allMatch(String.class::isInstance) ? list : null;
        if (value instanceof String text)
            return Arrays.stream(text.split(",", -1))
                    .map(String::strip)
                    .filter(part -> !part.isEmpty())
                    .toList();
        return value;
    }

    static boolean matches(String input, String operator, Object value) {
        if (operator.equals("empty")) return input.isEmpty();
        if (operator.equals("not empty")) return !input.isEmpty();
        if (operator.equals("in") || operator.equals("not in")) {
            if (!(value instanceof List<?> values)) return false;
            if (operator.equals("in"))
                return values.stream().anyMatch(item -> toStringValue(item).equals(input));
            return values.stream().noneMatch(item -> lower(toStringValue(item)).equals(lower(input)));
        }
        String comparison = toStringValue(value);
        return switch (operator) {
            case "contains" -> lower(input).contains(lower(comparison));
            case "not contains" -> !lower(input).contains(lower(comparison));
            case "start with" -> lower(input).startsWith(lower(comparison));
            case "end with" -> lower(input).endsWith(lower(comparison));
            default -> compareValues(input, comparison, operator);
        };
    }

    static boolean compareValues(String input, String value, String operator) {
        if (DATE.matcher(value).matches()) {
            if (!DATE.matcher(input).matches()) return operator.equals("≠");
            return compareStrings(input, value, operator);
        }
        Double left = numeric(input), right = numeric(value);
        if (left != null && right != null) {
            return switch (operator) {
                case "=" -> left.doubleValue() == right.doubleValue();
                case "≠" -> left.doubleValue() != right.doubleValue();
                case ">" -> left > right;
                case "<" -> left < right;
                case "≥" -> left >= right;
                case "≤" -> left <= right;
                default -> false;
            };
        }
        return compareStrings(lower(input), lower(value), operator);
    }

    private static Double numeric(String value) {
        if (value.equalsIgnoreCase("inf")
                || value.equalsIgnoreCase("+inf")
                || value.equalsIgnoreCase("infinity")
                || value.equalsIgnoreCase("+infinity")) return Double.POSITIVE_INFINITY;
        if (value.equalsIgnoreCase("-inf") || value.equalsIgnoreCase("-infinity"))
            return Double.NEGATIVE_INFINITY;
        if (value.equalsIgnoreCase("nan")) return Double.NaN;
        if (!FLOAT.matcher(value).matches()) return null;
        try {
            double result = Double.parseDouble(value.replace("_", ""));
            return Double.isInfinite(result) ? null : result;
        } catch (NumberFormatException error) {
            return null;
        }
    }

    private static boolean compareStrings(String left, String right, String operator) {
        int compared =
                Arrays.compareUnsigned(
                        left.getBytes(StandardCharsets.UTF_8), right.getBytes(StandardCharsets.UTF_8));
        return switch (operator) {
            case "=" -> compared == 0;
            case "≠" -> compared != 0;
            case ">" -> compared > 0;
            case "<" -> compared < 0;
            case "≥" -> compared >= 0;
            case "≤" -> compared <= 0;
            default -> false;
        };
    }

    private static String lower(String text) {
        StringBuilder result = new StringBuilder();
        text.codePoints()
                .forEach(codePoint -> result.appendCodePoint(Character.toLowerCase(codePoint)));
        return result.toString();
    }

    private static String toStringValue(Object value) {
        if (value instanceof String text) return text;
        if (value instanceof Number number) {
            double numeric = number.doubleValue();
            if (!Double.isFinite(numeric)) return Double.toString(numeric).replace("Infinity", "Inf");
            return BigDecimal.valueOf(numeric).stripTrailingZeros().toPlainString();
        }
        if (value instanceof Boolean bool) return bool.toString();
        return "";
    }

    private static String string(Object value) {
        return value instanceof String text ? text : "";
    }

    static List<String> constrainDocIds(List<String> base, List<String> filtered) {
        Set<String> allowed = new LinkedHashSet<>(filtered);
        if (base == null || base.isEmpty()) return new ArrayList<>(allowed);
        return base.stream().filter(allowed::contains).distinct().toList();
    }

    private static List<Condition> parseConditions(List<?> rows) {
        List<Condition> conditions = new ArrayList<>();
        for (Object row : rows)
            if (row instanceof Map<?, ?> map) {
                conditions.add(
                        new Condition(string(map.get("key")), map.get("value"), string(map.get("op"))));
            }
        return conditions;
    }

    static Generated generate(
            Map<String, Map<String, List<String>>> metadata,
            String question,
            Map<String, String> constraints,
            ChatPipeline.ChatModel model,
            String date)
            throws Exception {
        if (model == null) throw new IllegalArgumentException("chat model is nil");
        if (metadata.isEmpty()) return new Generated(List.of(), "and");
        Map<String, List<String>> structure = new TreeMap<>();
        metadata.forEach((key, values) -> structure.put(key, new ArrayList<>(values.keySet())));
        String constraintsJson =
                constraints == null ? "" : JSON.writeValueAsString(new TreeMap<>(constraints));
        String prompt =
                renderPrompt(date, JSON.writeValueAsString(structure), question, constraintsJson);
        ChatPipeline.ModelAnswer answer =
                model.complete(
                        List.of(
                                new ContextBuilder.Message("system", prompt),
                                new ContextBuilder.Message("user", "Generate filters:")),
                        Map.of());
        if (answer == null || answer.answer() == null)
            throw new IllegalStateException("empty response from meta filter generation");
        String response = answer.answer().strip();
        int think = response.lastIndexOf("</think>");
        if (think >= 0) response = response.substring(think + "</think>".length());
        response = response.replaceAll("```json\\n|```\\n*$", "").strip();
        Map<String, Object> parsed;
        try {
            parsed = JSON.readValue(response, new com.fasterxml.jackson.core.type.TypeReference<>() {
            });
        } catch (Exception invalid) {
            try {
                parsed =
                        JSON.readValue(
                                repairJson(response), new com.fasterxml.jackson.core.type.TypeReference<>() {
                                });
            } catch (Exception unrepaired) {
                return new Generated(List.of(), "and");
            }
        }
        if (parsed == null) return new Generated(List.of(), "");
        Map<String, Object> resultFields = structFields(parsed);
        if (resultFields.get("logic") != null && !(resultFields.get("logic") instanceof String))
            return new Generated(List.of(), "and");
        List<?> conditions = resultFields.get("conditions") instanceof List<?> list ? list : List.of();
        List<Condition> generated = new ArrayList<>();
        for (Object item : conditions) {
            if (item == null) {
                generated.add(new Condition("", null, ""));
                continue;
            }
            if (!(item instanceof Map<?, ?> condition)) return new Generated(List.of(), "and");
            Map<String, Object> fields = structFields(condition);
            if (fields.get("key") != null && !(fields.get("key") instanceof String)
                    || fields.get("op") != null && !(fields.get("op") instanceof String))
                return new Generated(List.of(), "and");
            generated.add(
                    new Condition(string(fields.get("key")), fields.get("value"), string(fields.get("op"))));
        }
        return new Generated(generated, string(resultFields.get("logic")));
    }

    // Go encoding/json 接受结构体字段名的大小写差异，手工 map 条件仍按原键读取。
    private static Map<String, Object> structFields(Map<?, ?> source) {
        Map<String, Object> result = new LinkedHashMap<>();
        source.forEach((key, value) -> result.put(key.toString().toLowerCase(Locale.ROOT), value));
        return result;
    }

    /**
     * 调用与原 Go 服务完全同版本的 jsonrepair.Repair，错误沿用原版空条件回退。
     */
    static String repairJson(String text) throws IOException, InterruptedException {
        String helper =
                System.getProperty("RAGFLOW_JSONREPAIR_HELPER", System.getenv("RAGFLOW_JSONREPAIR_HELPER"));
        if (helper == null || helper.isBlank()) helper = "native/build/ragflow-jsonrepair";
        Process process =
                new ProcessBuilder(helper).redirectError(ProcessBuilder.Redirect.DISCARD).start();
        try {
            try (var input = process.getOutputStream()) {
                input.write(text.getBytes(StandardCharsets.UTF_8));
            }
            byte[] repaired;
            try (var output = process.getInputStream()) {
                repaired = output.readAllBytes();
            }
            if (process.waitFor() != 0) throw new IOException("jsonrepair.Repair failed");
            return new String(repaired, StandardCharsets.UTF_8);
        } finally {
            if (process.isAlive()) process.destroyForcibly();
        }
    }

    static String renderPrompt(
            String date, String metadataJson, String question, String constraints) {
        String result =
                PROMPT
                        .replace("{{ current_date }}", date)
                        .replace("{{ metadata_keys }}", metadataJson)
                        .replace("{{ user_question }}", question);
        // 原 renderMetaFilterTemplate 只处理条件块，没有替换 {{ constraints }}。
        result = CONSTRAINT_BLOCK.matcher(result).replaceAll(constraints.isEmpty() ? "" : "$1");
        return result.replaceAll("\\n{3,}", "\n\n").strip();
    }
}
