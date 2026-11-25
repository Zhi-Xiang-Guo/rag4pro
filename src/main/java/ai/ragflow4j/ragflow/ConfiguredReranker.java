package ai.ragflow4j.ragflow;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.*;

/**
 * 显式配置的 query/documents rerank 协议适配，分数归一化由原版 Reranker 执行。
 */
public final class ConfiguredReranker implements Reranker.Model {
    private final URI endpoint;
    private final String key;
    private final String model;
    private final HttpClient http =
            HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    private final ObjectMapper json = new ObjectMapper();

    public ConfiguredReranker(URI endpoint, String key, String model) {
        this.endpoint = endpoint;
        this.key = key;
        this.model = model;
    }

    @Override
    public List<Reranker.ModelScore> rerank(String query, List<String> documents) throws Exception {
        var request =
                HttpRequest.newBuilder(endpoint)
                        .timeout(Duration.ofSeconds(90))
                        .header("Content-Type", "application/json");
        if (key != null && !key.isBlank()) request.header("Authorization", "Bearer " + key);
        var response =
                http.send(
                        request
                                .POST(
                                        HttpRequest.BodyPublishers.ofString(
                                                json.writeValueAsString(
                                                        Map.of("model", model, "query", query, "documents", documents))))
                                .build(),
                        HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200)
            throw new IllegalStateException("Reranker HTTP " + response.statusCode());
        var data = json.readTree(response.body()).path("results");
        if (!data.isArray()) throw new IllegalStateException("Reranker results 格式错误");
        List<Reranker.ModelScore> scores = new ArrayList<>();
        Set<Integer> seen = new HashSet<>();
        for (var row : data) {
            if (!row.path("index").canConvertToInt() || !row.path("relevance_score").isNumber())
                throw new IllegalStateException("Reranker 分数格式错误");
            int index = row.path("index").asInt();
            double score = row.path("relevance_score").asDouble();
            if (index < 0 || index >= documents.size() || !seen.add(index) || !Double.isFinite(score))
                throw new IllegalStateException("Reranker index 或分数无效");
            scores.add(new Reranker.ModelScore(index, score));
        }
        return scores;
    }
}
