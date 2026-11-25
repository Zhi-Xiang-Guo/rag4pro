/**
 * 标准文本 RAG 的算法与编排。建议按一条请求逐层阅读，而不是一次读完所有工具类。
 *
 * <p>检索主线：CorpusService.retrieve → StandardRetrieval.retrieve → QueryBuilder.question →
 * ElasticsearchStore.search → Reranker。前半段召回候选，后半段重新评分并选择最终结果。
 *
 * <p>建索引主线：CorpusService.index → GeneralChunker → IndexingPipeline → ElasticsearchStore。 文档先被拆成
 * chunk，再同时保存词法字段和 embedding 向量，供查询的两条匹配路径使用。
 *
 * <p>问答主线：CorpusService.answer → 检索与父块回填 → ContextBuilder → ChatPipeline → CitationService。
 * 检索返回证据，回答模型根据上下文生成文字，引用将文字关联到本次证据数组。
 *
 * <p>字段名保留原版约定以便对照 Go 源码；更完整的字段表、断点和参数关系见 docs/reading-guide.md。
 */
package ai.ragflow4j.ragflow;
