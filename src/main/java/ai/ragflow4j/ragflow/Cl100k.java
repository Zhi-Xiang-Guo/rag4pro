package ai.ragflow4j.ragflow;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingType;
import com.knuddels.jtokkit.api.IntArrayList;

import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * 与 internal/tokenizer 的 cl100k_base 使用相同 BPE，负责分块和标准上下文预算。
 */
public final class Cl100k implements ContextBuilder.TokenCounter {
    private final Encoding encoding =
            Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.CL100K_BASE);

    @Override
    public int count(String text) {
        return encoding.countTokensOrdinary(text);
    }

    public List<Integer> encode(String text) {
        IntArrayList encoded = encoding.encodeOrdinary(text);
        List<Integer> result = new ArrayList<>(encoded.size());
        for (int index = 0; index < encoded.size(); index++) result.add(encoded.get(index));
        return result;
    }

    public String decode(List<Integer> tokens) {
        IntArrayList values = new IntArrayList(tokens.size());
        for (int value : tokens) values.add(value);
        return encoding.decode(values);
    }

    @Override
    public String trim(String text, int maxTokens) {
        if (maxTokens <= 0) return "";
        List<Integer> tokens = encode(text);
        if (tokens.size() <= maxTokens) return text;
        IntArrayList prefix = new IntArrayList(maxTokens);
        for (int token : tokens.subList(0, maxTokens)) prefix.add(token);
        byte[] bytes = encoding.decodeBytes(prefix);
        String trimmed = clampUtf8(bytes);
        for (int i = 0; i < 4 && !trimmed.isEmpty() && count(trimmed) > maxTokens; i++) {
            trimmed = trimmed.substring(0, trimmed.offsetByCodePoints(trimmed.length(), -1));
        }
        return trimmed;
    }

    private static String clampUtf8(byte[] bytes) {
        for (int cut = 0; cut <= 4 && cut <= bytes.length; cut++) {
            try {
                return StandardCharsets.UTF_8
                        .newDecoder()
                        .decode(ByteBuffer.wrap(bytes, 0, bytes.length - cut))
                        .toString();
            } catch (CharacterCodingException ignored) {
                // BPE 截断点可能位于一个 Unicode 字符的 UTF-8 字节中间，需退回完整字符边界。
            }
        }
        return "";
    }
}
