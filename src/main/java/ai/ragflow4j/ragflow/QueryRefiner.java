// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations.
package ai.ragflow4j.ragflow;

import ai.ragflow4j.ragflow.ContextBuilder.Message;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.*;
import java.util.*;

/**
 * 借助回答模型为检索整理问题：多轮补全、关键词增强和跨语言转换。
 *
 * <p>对应 generator.go 的 FullQuestion、KeywordExtraction 和 CrossLanguages，使用原版模板。 这里产生的是检索输入文本，最终回答由
 * ChatPipeline 单独生成。
 */
public final class QueryRefiner {
    public record Outcome(String text, Exception error) {
    }

    private final ChatPipeline.ChatModel model;
    private final Clock clock;

    public QueryRefiner(ChatPipeline.ChatModel model) {
        this(model, Clock.system(ZoneId.of("Asia/Shanghai")));
    }

    public QueryRefiner(ChatPipeline.ChatModel model, Clock clock) {
        this.model = Objects.requireNonNull(model);
        this.clock = Objects.requireNonNull(clock);
    }

    public String fullQuestion(List<Message> messages, String language) {
        return fullQuestionOutcome(messages, language).text();
    }

    public Outcome fullQuestionOutcome(List<Message> messages, String language) {
        if (messages.isEmpty())
            return new Outcome("", new IllegalArgumentException("FullQuestion: empty messages"));
        String fallback = latestUser(messages);
        String conversation =
                String.join(
                        "\n",
                        messages.stream()
                                .filter(m -> m.role().equals("user") || m.role().equals("assistant"))
                                .map(m -> m.role().toUpperCase(Locale.ROOT) + ": " + m.content())
                                .toList());
        Instant now = clock.instant();
        String template = load("full_question");
        int start = template.indexOf("{{- if .Language }}");
        int end = template.indexOf("{{- end }}", start) + "{{- end }}".length();
        String requirement =
                language == null || language.isEmpty()
                        ? "\n- Text generated MUST be in the same language as the original user's question."
                        : "\n- Text generated MUST be in " + language + ".";
        // Go {{- consumes preceding whitespace, including the preceding newline.
        String prefix = template.substring(0, start).stripTrailing();
        template = prefix + requirement + template.substring(end);
        String system =
                template
                        .replace("{{.Today}}", date(now))
                        .replace("{{.Yesterday}}", date(now.minus(Duration.ofHours(24))))
                        .replace("{{.Tomorrow}}", date(now.plus(Duration.ofHours(24))))
                        .replace("{{.Conversation}}", conversation);
        try {
            String text =
                    answer(
                            model.complete(
                                    List.of(new Message("system", system), new Message("user", "Output: ")),
                                    Map.of()));
            text = stripThink(text.strip()).strip();
            if (text.isEmpty() || text.contains("**ERROR**")) return new Outcome(fallback, null);
            return new Outcome(text, null);
        } catch (Exception e) {
            return new Outcome(fallback, e);
        }
    }

    public String keywords(String content, int topN) throws Exception {
        if (content.isEmpty()) return "";
        if (topN <= 0) topN = 3;
        String prompt =
                load("keyword_prompt")
                        .replace("{{ topn }}", Integer.toString(topN))
                        .replace("{{ content }}", content);
        String response =
                answer(
                        model.complete(
                                List.of(new Message("system", prompt), new Message("user", "Output: ")),
                                Map.of("temperature", 0.2)));
        String result = stripThink(response.strip()).strip();
        if (result.contains("**ERROR**"))
            throw new IllegalStateException("error in keyword extraction response");
        return result;
    }

    public String crossLanguages(String query, List<String> languages) {
        return crossLanguagesOutcome(query, languages).text();
    }

    public Outcome crossLanguagesOutcome(String query, List<String> languages) {
        if (query.isEmpty() || languages.isEmpty()) return new Outcome(query, null);
        String prompt =
                load("cross_languages_user_prompt")
                        .replace("{{ query }}", query)
                        .replace("{{ languages | join(', ') }}", String.join(", ", languages));
        try {
            String result =
                    answer(
                            model.complete(
                                    List.of(
                                            new Message("system", load("cross_languages_sys_prompt")),
                                            new Message("user", prompt)),
                                    Map.of("temperature", 0.2)));
            result = stripThink(result);
            if (result.contains("**ERROR**")) return new Outcome(query, null);
            result = result.replaceFirst("(?i)^output:\\s*", "").replaceAll("\n+", "");
            List<String> translations =
                    Arrays.stream(result.split("===", -1))
                            .map(String::strip)
                            .filter(s -> !s.isEmpty())
                            .toList();
            return new Outcome(translations.isEmpty() ? query : String.join("\n", translations), null);
        } catch (Exception e) {
            return new Outcome(query, e);
        }
    }

    public String appendKeywords(String question) {
        try {
            return appendKeywords(question, keywords(question, 3));
        } catch (Exception ignored) {
            return question;
        }
    }

    public static String appendKeywords(String question, String keywords) {
        return keywords.isEmpty() ? question : question + "," + keywords;
    }

    public static String latestUser(List<Message> messages) {
        for (int i = messages.size() - 1; i >= 0; i--)
            if (messages.get(i).role().equals("user")) return messages.get(i).content();
        return "";
    }

    private String date(Instant instant) {
        return instant.atZone(clock.getZone()).toLocalDate().toString();
    }

    private static String answer(ChatPipeline.ModelAnswer response) {
        if (response == null || response.answer() == null)
            throw new IllegalStateException("empty response from query refinement");
        return response.answer();
    }

    static String stripThink(String text) {
        int end = text.lastIndexOf("</think>");
        return end < 0 ? text : text.substring(end + "</think>".length());
    }

    private static String load(String name) {
        try (var stream = QueryRefiner.class.getResourceAsStream("/ragflow/prompts/" + name + ".md")) {
            if (stream == null) throw new IllegalStateException("Missing prompt " + name);
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot read prompt " + name, e);
        }
    }
}
