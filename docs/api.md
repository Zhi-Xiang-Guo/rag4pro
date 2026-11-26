# API 与调试

服务启动后，以下命令可直接使用。

```bash
curl -fsS http://127.0.0.1:18080/api/health
curl -fsS -X POST http://127.0.0.1:18080/api/demo
curl -fsS http://127.0.0.1:18080/api/demo/chat \
  -H 'Content-Type: application/json' \
  -d '{"version":"v2","question":"支付通知超时后，平台多久第一次补发？"}'
curl -fsS 'http://127.0.0.1:18080/api/demo/source?docId=notify-v2'
```

`GET /api/demo` 返回手册版本对应的示例问题和索引状态。重复调用 `POST /api/demo` 会检查文件标题、元数据和内容 SHA-256，再补齐缺失项。问答应用 v1 或 v2 的文档过滤，返回 `answerStatus`、`reference`、`retrievalTrace` 和模型选择诊断。配了 `EVIDENCE_RAG_HTTP_TOKEN` 时，业务 API 需要 `Authorization: Bearer <token>`，演示页的连接设置可填同一个令牌。

## 知识工作台

- `GET /api/datasets` 列出知识库和文档数，包含原有 API 创建的数据集。`POST /api/datasets` 保存 `datasetId`、`name`、`language` 和默认分块配置。`DELETE /api/datasets?datasetId=...` 删除空库。
- `POST /api/chunk-preview` 用 `datasetId`、`title`、`fileType`、`text` 和分块配置预览，不调用模型。单篇与批量建索引共用这套分块逻辑。
- `chunkStrategy` 支持 `general`、`fixed`、`section`、`parent_child`、`hierarchical`。显式请求优先于知识库默认值。新策略的 `chunkTokenSize` 支持 32–1024，固定窗口还可以设置 `overlapPercent`。
- `POST /api/retrieve/context` 返回原始命中 `hits`、预算内上下文 `evidence`、正文 tokens、召回耗时和轨迹。它与 `/api/chat` 共用 `contextMode=leaf|parent|section` 及 `evidenceBudget`。非零预算限制证据正文的 Cl100k token 数。一个章节或父段找不到或放不进剩余预算时，该组命中逐级回退到父段和子块，再处理其余候选。

## 通用索引与检索

```bash
curl -fsS http://127.0.0.1:18080/api/documents \
  -H 'Content-Type: application/json' --data-binary @samples/shop-policy.json

curl -fsS http://127.0.0.1:18080/api/retrieve \
  -H 'Content-Type: application/json' --data-binary @samples/query.json

curl -fsS http://127.0.0.1:18080/api/chat \
  -H 'Content-Type: application/json' --data-binary @samples/query.json

curl -fsS 'http://127.0.0.1:18080/api/chunks?datasetId=demo-standard&docId=shop-policy'
curl -fsS 'http://127.0.0.1:18080/api/documents?datasetId=demo-standard'
```

- `POST /api/documents` 使用 `datasetId`、`docId`、`title`、`text` 建立或更新文档。支持 `fileType` 为 txt/md，默认分块预算为 512 token、无重叠、标题向量权重为 0.1。每批最多 64 篇新文档可通过 `POST /api/documents/batch` 一次写入。
- `POST /api/retrieve` 使用 `datasetId`、`question`。用 `strategy=standard|lexical|dense|rrf|rrf_diverse` 选择检索策略，`rrf_diverse` 优先返回不同文档各自排名最高的块。`topK` 控制输出数量，`candidateK` 控制总候选预算，`knnTopK`、`knnNumCandidates` 控制 ANN 搜索。标准分支支持 `mode=hybrid|bm25|vector`、评分阈值和排名特征。对于 RRF，分数来自名次，余弦阈值不适用。
- `docId`、`docIds`、`metadataFilter` 约束文档集合。示例请求采用 `{"method":"manual","manual":[{"key":"version","op":"=","value":"v2"}]}`。元数据筛选也支持 `auto` 和 `semi_auto`。
- `POST /api/chat` 默认走标准问答，支持对话 `messages`、`quote`、`refineMultiturn`、`crossLanguages`、`keyword`、`systemPrompt`、`emptyResponse` 和 `generation`。
- `answerMode=extractive` 先让模型形成简短回答草稿并选择最多三段原文，Java 按经过校验的证据编号渲染摘录。返回 `reference.spans`，其中 `citationIndex` 对应 `reference.chunks` 的下标，`[start,end)` 是相对 `content` 的 UTF-16 位置。每个文档同时提供外部文档 ID。设置 `evaluationTrace=true` 可以查看检索片段、模型输入与所选编号。
- 证据摘录可设置 `refineMultiturn=true`。服务用最近的用户问题补齐当前追问的指代，按独立问题检索和选证据。改写不可用时直接使用当前问题。
- `503 local_model_unavailable` 表示本地回答模型连接失败或拒绝请求。瞬时连接和服务错误可重试。证据格式出错时，服务会改用仅含编号的格式尝试纠正一次。整个证据选择共用 30 秒预算。`502 invalid_evidence_selection` 表示纠正后的证据选择仍未通过校验。`429` 带 `Retry-After`，可在请求数下降后重试。

需要第三方重排服务时配置 `EVIDENCE_RAG_RERANK_URL`、`EVIDENCE_RAG_RERANK_MODEL`、`EVIDENCE_RAG_RERANK_API_KEY`，在标准检索请求中传 `useReranker=true`。

`EVIDENCE_RAG_MODEL` 和嵌入服务 URL 都参与索引身份。切换嵌入配置后，为新数据建立独立数据集。在宿主机直接运行 Java 时，原 `RAGFLOW4J_` 环境变量可作同名设置的后备配置。
