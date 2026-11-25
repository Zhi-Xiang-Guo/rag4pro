// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations.
package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.text.Normalizer;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Qwen embedding 模型的 BPE 编码、计数和裁剪器，移植自 internal/tokenizer/bpe.go。
 *
 * <p>用于控制发给 embedding 模型的输入长度；分块与聊天预算仍使用 Cl100k。
 */
public final class QwenBpe implements ContextBuilder.TokenCounter {
    public static final String ASSET_SHA1 = "e6592f4d8d1678e963da9188090ac3eee916ed6b";
    private static final Pattern PIECES =
            Pattern.compile(
                    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r"
                            + "\\n"
                            + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                            + "\\n"
                            + "]*|\\s*[\\r"
                            + "\\n"
                            + "]+|\\s+(?!\\S)|\\s+",
                    Pattern.UNICODE_CHARACTER_CLASS);
    private final Map<String, Integer> vocabulary;
    private final Map<String, Integer> ranks;
    private final String[] alphabet = new String[256];
    private final Map<String, List<Integer>> cache =
            Collections.synchronizedMap(
                    new LinkedHashMap<>(256, 0.75f, true) {
                        @Override
                        protected boolean removeEldestEntry(Map.Entry<String, List<Integer>> eldest) {
                            return size() > 4096;
                        }
                    });

    public QwenBpe() {
        try (var stream =
                     QwenBpe.class.getResourceAsStream("/ragflow/tokenizers/qwen_bpe_tokenizer.json")) {
            if (stream == null) throw new IllegalStateException("Qwen tokenizer asset missing");
            byte[] bytes = stream.readAllBytes();
            String hash = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-1").digest(bytes));
            if (!hash.equals(ASSET_SHA1))
                throw new IllegalStateException("Qwen tokenizer asset hash mismatch");
            JsonNode root = new ObjectMapper().readTree(bytes);
            if (!root.path("model").path("type").asText().equals("BPE"))
                throw new IllegalStateException("Qwen tokenizer must be BPE");
            Map<String, Integer> vocab = new HashMap<>();
            root.path("model")
                    .path("vocab")
                    .fields()
                    .forEachRemaining(e -> vocab.put(e.getKey(), e.getValue().asInt()));
            Map<String, Integer> merges = new HashMap<>();
            int rank = 0;
            for (JsonNode merge : root.path("model").path("merges")) {
                String pair =
                        merge.isArray() ? merge.get(0).asText() + " " + merge.get(1).asText() : merge.asText();
                merges.put(pair, rank++);
            }
            if (vocab.isEmpty() || merges.isEmpty())
                throw new IllegalStateException("Qwen vocabulary/merges missing");
            vocabulary = Map.copyOf(vocab);
            ranks = Map.copyOf(merges);
            int replacement = 256;
            for (int value = 0; value < 256; value++) {
                boolean visible =
                        value >= 33 && value <= 126 || value >= 161 && value <= 172 || value >= 174;
                alphabet[value] = Character.toString(visible ? value : replacement++);
            }
        } catch (Exception e) {
            throw new IllegalStateException("Cannot load exact Qwen BPE tokenizer", e);
        }
    }

    @Override
    public int count(String text) {
        return encode(text).size();
    }

    public List<Integer> encode(String text) {
        List<Integer> result = new ArrayList<>();
        var matcher = PIECES.matcher(Normalizer.normalize(text, Normalizer.Form.NFC));
        while (matcher.find()) result.addAll(piece(matcher.group()));
        return result;
    }

    @Override
    public String trim(String text, int limit) {
        if (limit <= 0) return "";
        if (count(text) <= limit) return text;
        int low = 0, high = text.codePointCount(0, text.length());
        while (low < high) {
            int middle = (low + high + 1) / 2;
            if (count(text.substring(0, text.offsetByCodePoints(0, middle))) <= limit) low = middle;
            else high = middle - 1;
        }
        return text.substring(0, text.offsetByCodePoints(0, low));
    }

    private record Pair(int rank, int left, int right, int leftVersion, int rightVersion) {
    }

    private List<Integer> piece(String text) {
        List<Integer> cached = cache.get(text);
        if (cached != null) return cached;
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int size = bytes.length;
        String[] symbols = new String[size];
        int[] previous = new int[size], next = new int[size], version = new int[size];
        boolean[] alive = new boolean[size];
        for (int i = 0; i < size; i++) {
            symbols[i] = alphabet[bytes[i] & 255];
            previous[i] = i - 1;
            next[i] = i + 1;
            alive[i] = true;
        }
        if (size == 0) return List.of();
        next[size - 1] = -1;
        PriorityQueue<Pair> heap =
                new PriorityQueue<>(Comparator.comparingInt(Pair::rank).thenComparingInt(Pair::left));
        for (int i = 0; i < size; i++) push(heap, symbols, next, version, i);
        List<Pair> deferred = new ArrayList<>();
        int round = -1;
        while (!heap.isEmpty() || !deferred.isEmpty()) {
            if (heap.isEmpty()) {
                heap.addAll(deferred);
                deferred.clear();
                round = -1;
            }
            Pair pair = heap.remove();
            int left = pair.left(), right = pair.right();
            if (!alive[left]
                    || !alive[right]
                    || version[left] != pair.leftVersion()
                    || version[right] != pair.rightVersion()
                    || next[left] != right) continue;
            if (round >= 0 && pair.rank() > round) {
                heap.add(pair);
                heap.addAll(deferred);
                deferred.clear();
                round = -1;
                continue;
            }
            if (round >= 0 && pair.rank() < round) {
                deferred.add(pair);
                continue;
            }
            round = pair.rank();
            symbols[left] += symbols[right];
            version[left]++;
            alive[right] = false;
            next[left] = next[right];
            if (next[right] >= 0) previous[next[right]] = left;
            push(heap, symbols, next, version, previous[left]);
            push(heap, symbols, next, version, left);
        }
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i >= 0; i = next[i]) {
            Integer id = vocabulary.get(symbols[i]);
            if (id == null) throw new IllegalStateException("Qwen BPE produced an unknown symbol");
            ids.add(id);
        }
        ids = List.copyOf(ids);
        cache.put(text, ids);
        return ids;
    }

    private void push(
            PriorityQueue<Pair> heap, String[] symbols, int[] next, int[] version, int left) {
        if (left < 0 || next[left] < 0) return;
        int right = next[left];
        Integer rank = ranks.get(symbols[left] + " " + symbols[right]);
        if (rank != null) heap.add(new Pair(rank, left, right, version[left], version[right]));
    }
}
