# 标准 RAG 链路对齐

用户要求保留原版核心算法和处理步骤，并补齐回答与引用。

基准为本地 ragflow 提交 e4bd0801d5169c3dc0847ce0ea06fc6c3e516105。当前工作区的中文注释变化不改变行为。

## 审计结果

现有 Java 实现与原版存在实质差异。Unicode 定长切块、CJK 双字分词、正文单独向量、Lucene 单字段 BM25、双路独立截取并集以及 BM25/(1+BM25) 融合均需替换。原版的父子块处理、模型重排、上下文预算、生成与引用还需移植。

## 实施清单

- [x] 使用原版 native RAGAnalyzer 和同一词典进行粗细分词、词频和词性查询。
- [x] 移植 QueryBuilder、TermWeightDealer 与同义词扩展。
- [x] 移植标准 GeneralChunker 的分隔、合并、overlap、children 行为及 cl100k 计数。
- [x] 移植 token 字段生成、模型长度处理和标题正文向量混合。
- [x] 按原版 ES mapping、召回请求和 KNN 二次评分执行检索。
- [x] 移植 TokenSimilarity、RerankStandard、RerankWithKNN、模型重排和 rank features。
- [x] 保留候选过滤、分页、阈值、父子块回填和出处字段。
- [x] 补齐标准问答的查询处理、上下文预算、生成与引用。
- [x] 将新链路接入 HTTP 与页面，并使用 Docker 验证。
- [x] 用原版来源的固定样本验证各阶段输出，记录未通过和未覆盖项。

用户此次选定标准 RAG 全链路。Agentic 策略与复杂解析扩展单独列在审计范围中，验收聚焦标准链路。
