package ai.ragflow4j.ragflow;

import java.util.*;

/** RRF 按名次融合，不把 BM25 和向量相似度当作同一种量纲。每个分支内重复 ID 只计一次。 */
public final class ReciprocalRankFusion {
    private ReciprocalRankFusion() {}

    public static Map<String, Double> scores(List<List<String>> rankings, int k) {
        if (k < 1) throw new IllegalArgumentException("RRF k must be positive");
        Map<String, Double> scores = new HashMap<>();
        for (List<String> ranking : rankings) {
            Set<String> seen = new HashSet<>();
            int rank = 0;
            for (String id : ranking) {
                if (!seen.add(id)) continue;
                scores.merge(id, 1.0 / (k + ++rank), Double::sum);
            }
        }
        return scores;
    }

    public static List<String> rank(List<List<String>> rankings, int k) {
        Map<String, Double> scores = scores(rankings, k);
        return scores.keySet().stream().sorted(Comparator.<String>comparingDouble(scores::get)
                .reversed().thenComparing(Comparator.naturalOrder())).toList();
    }
}
