// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations.
package ai.ragflow4j.ragflow;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 将知识块转为提示词文本，并在模型上下文预算内保留或裁剪对话消息。
 *
 * <p>对应 chat_pipeline.go 的 kbPrompt、formatPrompt 和 messageFitIn，保留原版阶段顺序。
 */
public final class ContextBuilder {
    public interface TokenCounter {
        int count(String text);

        String trim(String text, int maxTokens);
    }

    public record Message(String role, String content) {
        public Message {
            Objects.requireNonNull(role);
            Objects.requireNonNull(content);
        }
    }

    public record Fitted(List<Message> messages, int tokens) {
    }

    private final TokenCounter tokens;

    public ContextBuilder(TokenCounter tokens) {
        this.tokens = Objects.requireNonNull(tokens);
    }

    /**
     * 用本次 chunks 的下标生成 ID；标题、URL 和 metadata 一起进入知识文本。
     */
    public List<String> knowledge(List<Map<String, Object>> chunks, int maxTokens) {
        int used = 0;
        int count = 0;
        // 原版会保留首次越过 97% 边界的那一块；这里不是逐块严格拒绝超预算内容。
        for (Map<String, Object> chunk : chunks) {
            String content = text(chunk, "content_with_weight", "content");
            if (content.isEmpty()) continue;
            used += tokens.count(content);
            count++;
            if (maxTokens * 0.97 < used) break;
        }
        List<String> result = new ArrayList<>();
        for (int i = 0; i < Math.min(count, chunks.size()); i++) {
            Map<String, Object> chunk = chunks.get(i);
            String content = text(chunk, "content_with_weight", "content");
            if (content.isEmpty()) continue;
            StringBuilder block = new StringBuilder("\nID: " + i);
            block.append(node("Title", text(chunk, "docnm_kwd", "document_name")));
            block.append(node("URL", text(chunk, "url")));
            if (chunk.get("document_metadata") instanceof Map<?, ?> metadata) {
                metadata.entrySet().stream()
                        .sorted(Comparator.comparing(e -> e.getKey().toString()))
                        .forEach(e -> block.append(node(e.getKey().toString(), e.getValue())));
            }
            block.append("\n└── Content:\n").append(content);
            result.add(block.toString());
        }
        return List.copyOf(result);
    }

    public static String formatPrompt(String template, Map<String, Object> parameters) {
        String result = template;
        for (var parameter : parameters.entrySet()) {
            result = result.replace("{" + parameter.getKey() + "}", goString(parameter.getValue()));
        }
        return result.replace("{knowledge}", " ").replace("{quote}", " ");
    }

    /**
     * 全量消息超预算时，先保留 system 和最后一条；仍超限才按原规则裁剪两端内容。
     */
    public Fitted fitMessages(List<Message> messages, int maxTokens) {
        if (messages.isEmpty()) throw new IllegalArgumentException("messages is empty");
        if (maxTokens <= 0) maxTokens = 8192;
        int total = count(messages);
        if (total < maxTokens) return new Fitted(List.copyOf(messages), total);
        List<Message> result = new ArrayList<>();
        for (Message message : messages) {
            if (message.role().equals("system")) result.add(message);
        }
        if (messages.size() > 1) result.add(messages.getLast());
        total = count(result);
        if (total < maxTokens) return new Fitted(List.copyOf(result), total);
        int system = tokens.count(result.getFirst().content());
        int last = tokens.count(result.getLast().content());
        if (system + last <= 0) return new Fitted(List.copyOf(result), 0);
        if (result.size() == 1) {
            trim(result, 0, maxTokens);
        } else if ((double) system / (system + last) > 0.8) {
            int preserved = Math.min(last, maxTokens);
            trim(result, result.size() - 1, preserved);
            trim(result, 0, Math.max(0, maxTokens - preserved));
        } else {
            int preserved = Math.min(system, maxTokens);
            trim(result, 0, preserved);
            trim(result, result.size() - 1, Math.max(0, maxTokens - preserved));
        }
        return new Fitted(List.copyOf(result), count(result));
    }

    private void trim(List<Message> messages, int index, int limit) {
        Message old = messages.get(index);
        messages.set(index, new Message(old.role(), tokens.trim(old.content(), limit)));
    }

    public int count(List<Message> messages) {
        return messages.stream().mapToInt(m -> tokens.count(m.content())).sum();
    }

    public static String citationPrompt() {
        try (var stream =
                     ContextBuilder.class.getResourceAsStream("/ragflow/prompts/citation_prompt.md")) {
            if (stream == null) throw new IllegalStateException("Missing citation_prompt.md");
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8).strip()
                    + "\n\n"
                    + "IMPORTANT: The example IDs above (45, 46, 78, etc.) are illustrative only. Use the"
                    + " actual chunk IDs from the provided knowledge blocks. Cite each source INDIVIDUALLY as"
                    + " [ID:i][ID:j]; NEVER merge consecutive citations into a range such as [ID:i-j].";
        } catch (IOException e) {
            throw new IllegalStateException("Cannot load citation_prompt.md", e);
        }
    }

    static String text(Map<String, Object> values, String... keys) {
        for (String key : keys) {
            if (values.get(key) instanceof String value) return value;
        }
        return "";
    }

    private static String node(String key, Object value) {
        String line = goString(value);
        return line.isEmpty() ? "" : "\n├── " + key + ": " + line.replaceAll("\n+", " ");
    }

    private static String goString(Object value) {
        if (value == null) return "<nil>";
        if (value instanceof List<?> list) {
            return "[" + String.join(" ", list.stream().map(ContextBuilder::goString).toList()) + "]";
        }
        if (value instanceof Map<?, ?> map) {
            return "map["
                    + String.join(
                    " ",
                    map.entrySet().stream()
                            .sorted(Comparator.comparing(e -> e.getKey().toString()))
                            .map(e -> e.getKey() + ":" + goString(e.getValue()))
                            .toList())
                    + "]";
        }
        return value.toString();
    }
}
