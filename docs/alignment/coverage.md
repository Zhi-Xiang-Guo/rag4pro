# 原版对齐审计

本次以本地 RAGFlow 提交 `e4bd0801d5169c3dc0847ce0ea06fc6c3e516105` 的 Go 代码为基准。用户选定标准 RAG 全链路，包含回答与引用。

初版 Java 采用固定字符切块、CJK 双字分词、正文向量、嵌入式 Lucene 和 BM25 归一化融合。这些替换改变了原始算法，初版与原版存在实质差异。上述实现已从当前源码入口移除。

## 当前实现与验证边界

| 阶段 | 当前实现 | 验证 |
|---|---|---|
| txt 解析与 GeneralChunker | 原始分隔、OVER_CAP 合并、overlap、children/mom | 原版来源的固定样本 |
| Markdown | 段落、标题、列表、代码、HTML、表格网格与合并策略 | CommonMark/Jsoup 消费语义样本，全部 gomarkdown 扩展仍需逐项验证 |
| chunk token budget | cl100k BPE 与 UTF-8 安全裁剪 | Unicode、token 边界和预算样本 |
| 词法分词 | 同一版本 C++ RAGAnalyzer、词典、OpenCC、WordNet | native C ABI、Go WordNet oracle |
| 模型 token budget | 原版固定 Qwen BPE 资源、模型窗口余量、收缩和单输入隔离 | HuggingFace tokenizer 实际 token IDs、模拟上游限长错误 |
| 索引字段 | coarse/fine/title/question/keyword/summary 字段、顺序 | 原版来源样本 |
| 标题向量 | raw filename 单独编码，默认 0.1/0.9 线性混合 | 固定向量与真实网关 |
| 正文向量缓存 | 原始 key、未加权向量、7 天 TTL、任务清理与失败重用 | Go xxhash key oracle、故障/标题权重变化样本 |
| metadata 文档过滤 | manual/auto/semi_auto、原始比较运算、空结果范围语义 | 原版运算规则、缺失条件与修复后模型响应样本 |
| 查询 | 清洗、语言分支、词频/DF/NER/POS 权重、同义词、短语、字段 boost | 英文、中文、变音符及精度样本 |
| ES mapping / recall | 原 mapping，query_string，kNN 过滤快照，原始 fusion 行为 | 请求契约与真实 ES 8.11.3 |
| 二阶段评分 | KNN、标准 fallback、cross-encoder、标签/PageRank | 原版来源数值样本、失败分支、请求契约 |
| 结果处理 | 文档存活校验、稳定排序、阈值、分页、聚合、父子块回填 | 固定样本及真实子块/父块检索 |
| 查询改写 | 最近三个问题、多轮改写、关键词与跨语言 prompt | 原始模板和失败分支样本 |
| 上下文 | kbPrompt、metadata、97%/95% 预算、completion 约束 | 原版来源样本 |
| 回答 | 标准同步文本消息调用 | 本机 Ollama 实际调用 |
| 引用 | 编号修复、已有引用提取、向量补齐、句子 cosine 补引、references | 固定样本和真实接口结果 |

## 影响“完全一致”判断的条件

应固定模型、同一模型输出、字典版本、输入文本、文档 ID、索引内容、分片、参数和所选分支，再比较中间结果。当前已经验证选定标准文本路径的核心阶段，整个 RAGFlow 项目的所有输入与配置组合仍有未验证项。

Java 使用 CommonMark AST 和 Jsoup DOM。现有 Markdown 文本与表格规则已按实际 Go 消费逻辑移植，并用固定样本验证。gomarkdown 的全部扩展语法、图片下载及 OCR/media 路径仍属于剩余兼容性范围。

当前 HTTP 入口为本机单租户、单数据集范围的标准同步文本调用。原版业务数据库与任务队列在此由文档目录和同步调用承担。多租户授权、多 KB 联查、流式协议、外部工具、SQL/web/KG、DeepDOC 和 Agentic 需要各自的实现与验证。metadata 条件生成使用同版本 kaptinlin/jsonrepair 依赖辅助程序进行 JSON 修复。自动标签分类配置仍需单独集成。原版 rank features 和模型重排核心函数已保留，可通过显式配置调用。

## 已保留的容易误读之处

- GeneralChunker 和 strict TokenChunker 的合并边界不同。本项目当前入口使用标准模板的 GeneralChunker。
- 初次 ES 召回的 fusion 权重与最终二阶段 vectorWeight 独立。
- ES KNN `_score` 与本地 fallback 的原始 cosine 数值含义不同，移植保留该分支行为。
- 父块的排序分数来自命中的子块平均值。父块可能保留零向量占位，引用补偿仍沿用原始向量可用性处理。
- citation ID 是上下文数组下标，原始 chunk ID 用于存储和取回。
- 当前资源目录和本地 Go 源码均没有 `term.freq` 文件，DF 路径保留原版缺文件时的行为。

## 回答模型的真实限制

本机现有 `qwen2.5:0.5b` 在真实测试中出现过把“三个工作日”回答成“3–5 个工作日”、误读英文系统提示词、忽略引用格式、引用块选择错误等现象。用户最终选择保留本地回答模型。检索、上下文、模型输出和引用步骤均可以分别检查，端到端测试验证链路与引用数据结构，回答事实准确性仍需逐条评估。

## 2026-11-24：测评扩展与原版边界

此次新增 `strategy=lexical/dense/rrf` 是用于消融和优化实验的扩展，不属于原版等价移植。未传 strategy 时仍使用上述 standard 流程；原有 mode 语义没有替换。RRF 不套用原版相似度阈值，独立 dense 不附词法前置过滤，两者与 standard 的差异应在测评中明确说明。

批量新增入口复用相同 GeneralChunker、IndexingPipeline 与 materialize，仅合并模型请求和持久化批次；增加单篇/批次字段与向量一致性测试，以及目录落盘失败后的补偿检查。query embedding 的有界 TTL cache 默认关闭，评测显式启用并标注预热条件。

回答增加可选 `evaluationTrace`，保存本次实际 model.complete 边界的 messages/generation/rawAnswer。它用于定位证据和引用问题，不改变标准生成提示词。证据优先提示词只作为测评候选，没有自动成为默认产品行为。

当前新增的 API token 是服务级准入，不能视作原版多租户授权的实现。整体生产验收状态与公开检索成绩分别记录在 `evaluation/reports/2026-11-24/`。
