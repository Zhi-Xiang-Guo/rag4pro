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
import java.io.InputStream;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 消费已检索并补齐字段的知识块，完成上下文组装、模型调用和引用整理。
 *
 * <p>对应标准 AsyncChat 的 10–11 阶段；检索与父块回填已由 CorpusService 完成。 阅读主线为 complete → prepare →
 * model.complete → CitationService.decorate。
 */
public final class ChatPipeline {
    @FunctionalInterface
    public interface ChatModel {
        ModelAnswer complete(List<Message> messages, Map<String, Object> generation) throws Exception;

        default ModelAnswer stream(List<Message> messages, Map<String, Object> generation, Stream output) throws Exception {
            throw new UnsupportedOperationException("聊天模型不支持流式回答");
        }
    }

    public record ModelAnswer(String answer, String reasoning, Map<String, Object> usage) {
        public ModelAnswer(String answer) {
            this(answer, "", Map.of());
        }
    }

    public record Parameter(String key, boolean optional) {
    }

    public record Request(
            List<Message> messages,
            String systemPrompt,
            List<Parameter> parameters,
            Map<String, Object> variables,
            Map<String, Object> generation,
            int modelMaxTokens,
            String emptyResponse,
            boolean quote,
            boolean storeHistoryMessages,
            String textAttachments,
            boolean hasImageAttachments) {
        public Request {
            messages = List.copyOf(messages);
            parameters =
                    parameters == null ? List.of(new Parameter("knowledge", false)) : List.copyOf(parameters);
            variables = variables == null ? Map.of() : Map.copyOf(variables);
            generation = generation == null ? Map.of() : Map.copyOf(generation);
            systemPrompt = Objects.requireNonNullElse(systemPrompt, "");
            textAttachments = Objects.requireNonNullElse(textAttachments, "");
            emptyResponse = Objects.requireNonNullElse(emptyResponse, "");
            if (modelMaxTokens <= 0) modelMaxTokens = 8192;
        }
    }

    public record Retrieval(List<Map<String, Object>> chunks, List<Map<String, Object>> docAggs) {
        public Retrieval {
            chunks = List.copyOf(chunks);
            docAggs = List.copyOf(docAggs);
        }
    }

    public record Prepared(
            List<Message> messages,
            List<String> knowledge,
            int promptTokens,
            Map<String, Object> generation,
            String prompt) {
    }

    public record Result(
            String answer,
            String reasoning,
            Map<String, Object> reference,
            Map<String, Object> usage,
            String prompt,
            List<Integer> citedIndices,
            boolean emptyResponse,
            Map<String, Object> audit) {
    }

    private final ChatModel model;
    private final ContextBuilder.TokenCounter tokens;
    private final ContextBuilder context;
    private final CitationService.Embedder embedder;
    private final CitationService.VectorHydrator hydrator;

    public ChatPipeline(
            ChatModel model,
            ContextBuilder.TokenCounter tokens,
            CitationService.Embedder embedder,
            CitationService.VectorHydrator hydrator) {
        this.model = Objects.requireNonNull(model);
        this.tokens = Objects.requireNonNull(tokens);
        this.context = new ContextBuilder(tokens);
        this.embedder = embedder;
        this.hydrator = hydrator;
    }

    public interface Stream {
        void delta(String text) throws IOException;
        void attach(InputStream upstream) throws IOException;
    }

    public Result complete(Request request, Retrieval retrieval) throws Exception {
        return complete(request, retrieval, null);
    }

    public Result complete(Request request, Retrieval retrieval, Stream stream) throws Exception {
        validate(request);
        // 先按原版 97% 规则选知识块；下方 prepare 还会对完整消息再次做预算处理。
        List<String> knowledge = context.knowledge(retrieval.chunks(), request.modelMaxTokens());
        if (knowledge.isEmpty()
                && request.textAttachments().isEmpty()
                && !request.hasImageAttachments()
                && !request.emptyResponse().isEmpty()) {
            Map<String, Object> reference = new LinkedHashMap<>();
            if (request.quote()) {
                reference.put("total", retrieval.chunks().size());
                reference.put("chunks", retrieval.chunks());
                reference.put("doc_aggs", retrieval.docAggs());
            }
            return new Result(
                    request.emptyResponse(),
                    "",
                    reference,
                    Map.of(),
                    "\n\n### Query:\n" + request.messages().getLast().content(),
                    List.of(),
                    true,
                    Map.of("modelInvoked", false, "messages", List.of()));
        }
        // 没走空知识固定回复时，才真正请求回答模型。
        Prepared prepared = prepare(request, knowledge);
        long modelStarted = System.nanoTime();
        ModelAnswer response = stream == null ? model.complete(prepared.messages(), prepared.generation())
                : model.stream(prepared.messages(), prepared.generation(), stream);
        double modelMs = (System.nanoTime() - modelStarted) / 1e6;
        long citationStarted = System.nanoTime();
        if (response == null || response.answer() == null)
            throw new IllegalStateException("Chat model returned no answer");
        // 引用编号属于本次知识块数组。此处修复模型的编号格式，必要时按向量相似度补引。
        CitationService.Decorated decorated =
                CitationService.decorate(
                        response.answer(),
                        retrieval.chunks(),
                        retrieval.docAggs(),
                        !knowledge.isEmpty(),
                        request.quote(),
                        embedder,
                        hydrator);
        Map<String, Object> usage = response.usage();
        if (usage == null || usage.isEmpty()) {
            int completionTokens = tokens.count(decorated.answer());
            usage =
                    Map.of(
                            "prompt_tokens",
                            prepared.promptTokens(),
                            "completion_tokens",
                            completionTokens,
                            "total_tokens",
                            prepared.promptTokens() + completionTokens);
        }
        return new Result(
                decorated.answer(),
                Objects.requireNonNullElse(response.reasoning(), ""),
                decorated.reference(),
                usage,
                prepared.prompt(),
                decorated.citedIndices(),
                false,
                Map.of("modelInvoked", true, "messages", prepared.messages(),
                        "generation", prepared.generation(), "promptTokensEstimated", prepared.promptTokens(),
                        "rawAnswer", response.answer(), "modelMs", modelMs,
                        "citationMs", (System.nanoTime() - citationStarted) / 1e6));
    }

    /**
     * 把知识、模板变量和历史组装为消息，并为回答留出模型上下文空间。
     */
    public Prepared prepare(Request request, List<String> knowledge) {
        validate(request);
        Map<String, Object> variables = new LinkedHashMap<>(request.variables());
        variables.put("date", Instant.now().toString());
        String template = request.systemPrompt();
        for (Parameter parameter : request.parameters()) {
            if (parameter.key().equals("knowledge")) continue;
            if (!variables.containsKey(parameter.key())) {
                if (!parameter.optional())
                    throw new IllegalArgumentException("Miss parameter: " + parameter.key());
                template = template.replace("{" + parameter.key() + "}", " ");
            }
        }
        String joined = String.join("\n\n------\n\n", knowledge);
        if (!joined.isEmpty()) variables.put("knowledge", "\n------\n" + joined);
        String system = ContextBuilder.formatPrompt(template, variables);
        if (!knowledge.isEmpty() && !template.contains("{knowledge}"))
            system += variables.get("knowledge");
        String citations =
                !knowledge.isEmpty() && request.quote() ? ContextBuilder.citationPrompt() : "";
        List<Message> messages = new ArrayList<>();
        messages.add(new Message("system", system));
        for (Message message : request.messages()) {
            if (message.role().equals("system") && request.storeHistoryMessages()) continue;
            messages.add(new Message(message.role(), CitationService.cleanHistory(message.content())));
        }
        if (!request.textAttachments().isEmpty()) {
            Message last = messages.getLast();
            messages.set(
                    messages.size() - 1,
                    new Message(last.role(), last.content() + request.textAttachments()));
        }
        // 95% 是完整消息的预算；引用说明按原版顺序在裁剪后追加，随后重新计入总占用。
        ContextBuilder.Fitted fitted =
                context.fitMessages(messages, (int) (request.modelMaxTokens() * 0.95));
        if (fitted.messages().size() < 2) throw new IllegalStateException("message_fit_in has bug");
        int used = fitted.tokens() + tokens.count(citations);
        if (used >= request.modelMaxTokens()) {
            throw new IllegalArgumentException(
                    "prompt uses "
                            + used
                            + " tokens, leaving no completion capacity for model max context length "
                            + request.modelMaxTokens());
        }
        Map<String, Object> generation = new LinkedHashMap<>(request.generation());
        if (generation.get("max_tokens") instanceof Number maxTokens) {
            if (maxTokens.intValue() <= 0) generation.remove("max_tokens");
            else
                generation.put(
                        "max_tokens", Math.min(maxTokens.intValue(), request.modelMaxTokens() - used));
        }
        messages = new ArrayList<>(fitted.messages());
        String prompt = messages.getFirst().content();
        messages.set(0, new Message("system", prompt + citations));
        return new Prepared(List.copyOf(messages), List.copyOf(knowledge), used, generation, prompt);
    }

    private static void validate(Request request) {
        if (request.messages().isEmpty())
            throw new IllegalArgumentException("AsyncChat: messages is empty");
        if (!request.messages().getLast().role().equals("user")) {
            throw new IllegalArgumentException("the last content of this conversation is not from user");
        }
    }
}
