package ai.ragflow4j;

import java.util.List;

/**
 * 文本向量化接口；调用者依赖向量结果，不需要了解模型 HTTP 协议。
 */
public interface Embeddings {
    String identity();

    List<double[]> embed(List<String> texts) throws Exception;

    /**
     * 与 OpenAIModel.Embed 一样直接发送查询原文，不额外拼接 query 指令前缀。
     */
    default double[] query(String question) throws Exception {
        return embed(List.of(question)).getFirst();
    }
}
