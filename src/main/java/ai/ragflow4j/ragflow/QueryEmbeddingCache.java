package ai.ragflow4j.ragflow;

import java.time.Duration;
import java.util.*;
import java.util.function.LongSupplier;

/** 单个模型实例内的有界 TTL 缓存；不写磁盘，key 不包含凭据，返回副本防止调用者污染。 */
public final class QueryEmbeddingCache implements StandardRetrieval.Embedding {
    private record Entry(double[] vector, long created) {}
    private final StandardRetrieval.Embedding delegate;
    private final int capacity;
    private final long ttl;
    private final LongSupplier clock;
    private final LinkedHashMap<String, Entry> entries = new LinkedHashMap<>(16, .75f, true);

    public QueryEmbeddingCache(StandardRetrieval.Embedding delegate, int capacity, Duration ttl) {
        this(delegate, capacity, ttl, System::nanoTime);
    }
    QueryEmbeddingCache(StandardRetrieval.Embedding delegate, int capacity, Duration ttl, LongSupplier clock) {
        if (capacity < 0 || ttl.isNegative() || ttl.isZero()) throw new IllegalArgumentException("invalid query cache limits");
        this.delegate = delegate;
        this.capacity = capacity;
        this.ttl = ttl.toNanos();
        this.clock = clock;
    }
    public synchronized double[] query(String text) throws Exception {
        if (capacity == 0) return delegate.query(text);
        long now = clock.getAsLong();
        Entry entry = entries.get(text);
        if (entry != null && now - entry.created() < ttl) return entry.vector().clone();
        // 失败不缓存；只有成功向量进入缓存。同一个实例绑定同一个 embedding 模型。
        double[] vector = delegate.query(text);
        entries.put(text, new Entry(vector.clone(), clock.getAsLong()));
        while (entries.size() > capacity) entries.remove(entries.keySet().iterator().next());
        return vector.clone();
    }
}
