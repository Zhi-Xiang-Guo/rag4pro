# Evidence RAG 评测

此目录实现“冻结数据 → 入库 → 开发集对照 → 冻结候选 → 留出测试 → 门禁 → 逐题诊断 → 面试证据”。结果以本次生成的 artifacts 为准。它是面向上线验收的测评基础，并不代表系统已经达到企业生产验收标准。

## 分块与证据回填

`compare_chunking.py` 从 HotpotQA distractor 官方验证集固定划出 24 道校准题和 100 道留出题。每题只检索官方提供的候选文章，五组配置共用同一个网关向量模型、候选池 32、最终 Top 8 和证据正文 512 tokens。金标支持句只用于评分。脚本会保留每题排序、完整支持句和用量，并校验入库语料指纹。

```bash
# 先按下方运行步骤配置网关及 Python 环境。
EVIDENCE_RAG_EVAL_PROJECT=rag4pro-hotpot EVIDENCE_RAG_EVAL_HTTP_PORT=18082 \
  docker compose -f compose.yaml -f evaluation/compose.eval.yaml up -d --build --wait
mkdir -p evaluation/data/hotpotqa
curl -fL 'https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet' \
  -o evaluation/data/hotpotqa/dev-distractor.parquet
.venv-eval/bin/python evaluation/compare_chunking.py --url http://127.0.0.1:18082 \
  --source evaluation/data/hotpotqa/dev-distractor.parquet --split calibration \
  --out evaluation/runs/rag4pro-hotpot/calibration
.venv-eval/bin/python evaluation/compare_chunking.py --url http://127.0.0.1:18082 \
  --source evaluation/data/hotpotqa/dev-distractor.parquet --split holdout \
  --out evaluation/runs/rag4pro-hotpot/holdout
```

语料文件 SHA-256 为 `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`。上面的 HotpotQA 服务和 SciFact 使用不同 Compose 项目，分别持有索引与文档目录。评测 Compose 配置已开启查询向量缓存。复现耗时时使用同一台机器运行各策略。`retrievalMs` 计量召回阶段，`requestMs` 计量整个请求。[HotpotQA 数据与论文入口](https://hotpotqa.github.io/)

## 支付手册留出集

支付手册评测使用与浏览器相同的 `/api/demo/chat`。它通过 API 建立索引、校验示例原文和 SHA-256，金标只用于评分。先跑 12 道开发题，冻结源码、题集、语料、模型和验收脚本，再运行 36 道留出题。

```bash
python3 evaluation/payment_qa.py --labels evaluation/payment-qa-v2.json --split dev --out evaluation/runs/payment/dev
python3 evaluation/payment_qa.py --labels evaluation/payment-qa-v2.json --split test --dev-summary evaluation/runs/payment/dev/summary.json \
  --out evaluation/runs/payment/test
```

`results.jsonl` 含逐题金标、实际引用、候选文档和错误。门禁要求有答案题的完整金标引用率和无答案题拒答率均至少 90%，所有题的版本及引用格式校验通过，且接口零错误。[本地实测报告](../docs/evaluation/payment.md)

## 运行

完整检索评测使用 Python 3.9+ 和 Docker Compose 2.24.4+。评测和服务共用项目根目录的 `.env`，按示例配置 OpenAI 兼容网关。入库前可调用 `/api/health` 核对网关向量模型、聊天模型及 Elasticsearch 索引。更换向量模型后用新的评测索引入库。[配置示例](../.env.example)

```bash
python3 -m venv .venv-eval
.venv-eval/bin/pip install -r evaluation/requirements.txt
.venv-eval/bin/python evaluation/eval.py prepare

docker compose -f compose.yaml -f evaluation/compose.eval.yaml up -d --build
# 独立 ES index / 数据卷，HTTP 18081；原有演示数据卷不覆盖。
python3 evaluation/eval.py ingest --url http://127.0.0.1:18081
python3 evaluation/eval.py run --url http://127.0.0.1:18081 --split dev --warmup --out evaluation/runs/dev
python3 evaluation/eval.py select --out evaluation/runs/dev
python3 evaluation/eval.py run --url http://127.0.0.1:18081 --split test --warmup \
  --selection evaluation/runs/dev/selection.json --out evaluation/runs/test
python3 evaluation/eval.py gate --out evaluation/runs/test

python3 evaluation/qa.py --url http://127.0.0.1:18081 --split dev --out evaluation/runs/qa-dev
python3 evaluation/qa.py --url http://127.0.0.1:18081 --split test \
  --selection evaluation/runs/qa-dev/selection.json --out evaluation/runs/qa-test
python3 evaluation/load.py --url http://127.0.0.1:18081 \
  --selection evaluation/runs/dev/selection.json --out evaluation/runs/load
# 整体门禁可能预期退出 1；失败原因保留在 release-gate.json。
python3 evaluation/release_gate.py
# 诊断重放在检索性能测试完成后单独运行，避免相互干扰。
python3 evaluation/replay.py
python3 evaluation/build_report.py
```

依赖下载只需运行一次。`prepare` 验证已有 manifest，不覆盖冻结数据。`ingest` 通过已有文档目录续传，校验来源哈希；批次失败会返回错误并停止，重跑即可继续。`run` 先校验全语料和版本，再按 `(queryId, variant)` 断点续跑；失败查询保留并记零分，不从分母移除。改变配置或源码后须新建 run 目录，不能拼接旧结果。测试阶段必须给出开发集产生的 selection。

## 数据和口径

- 公开集：[BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact)，完整 5,183 篇文献摘要；官方 train 中用哈希顺序固定选 60 条作开发集，官方 test 全部 300 条作留出测试。保留镜像 commit、原文件 SHA256、格式转换后的 SHA256、许可 CC-BY-SA-4.0 和来源。索引仅包含文档与来源元数据；问题和 qrels 不入索引。
- 它测量科学断言到文献的检索能力，不提供问答正确性或证据 span 标注。官方切分可能共享主题/相近断言，不能称为全新领域泛化。用于跨版本/租户的应用能力需另测。
- 指标是**文档级** Recall、MRR、nDCG、Precision @1/5/10。先按 chunk 排名去重文档，再截断到 k。零命中与请求失败均留在分母。nDCG 用 `(2^rel-1)/log2(rank+1)`；本集正相关标签均为 1。
- `standard` 为当前 Java 移植的原版默认流程；它不是官方 RAGFlow 完整服务的成绩。`lexical` 是独立词法路径，使用 RAGAnalyzer 和原 ES mapping/query，不声称是 vanilla BM25。原入口 `mode=bm25` 仅改变二阶段评分权重，仍会调用 embedding/kNN，不能作纯词法消融。
- `dense` 仅独立向量召回；`rrf` 将两路名次按 `sum(1/(60+rank))` 融合。RRF 总候选预算 64：两路各 32，其余组为 64；ANN 探索 k=256、numCandidates=1024。两路 ES 请求、词法处理等计算成本仍不同。
- RRF 分数不是 cosine，不能套用标准路径的 0.2 阈值；新路径显式不使用该阈值。`strategy` 默认仍为 `standard`。算法来源：[Cormack 等，SIGIR 2009](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf)。
- 排序质量逐题配对比较，附 2,000 次固定种子的 bootstrap 95% 区间及胜/平/负。区间假设查询独立；同主题相关性可能使区间偏窄。负结果和退化案例均输出。
- 检索运行显式预热每题向量，按种子打散组别顺序。P50/P95 是本机 Docker、温缓存、单客户端顺序请求的观察值；不包含冷 embedding 成本，不是生产 SLO。standard 为调试返回向量和更详细 DSL，新策略响应更小，延迟差异包含评分与序列化/传输成本，不能只归因于排序算法。另有独立有限并发探测，429 单列且包含在总请求中。

## 回答与引用

`qa-fixture.json` 是**人工编写但尚未人工复核的合成回归夹具**，不是真实公司政策。12 篇文档、28 条问题，按政策族分成 10 条 dev 和 18 条 test，包括多轮、版本 metadata 范围、同产品旧版本/其他产品干扰、无证据问题。不能将其小样本规则结果包装成业务准确率。

比较原提示词与证据优先提示词，仅在 dev 选优。保留同一次请求的模型最终 messages、generation、原始回答、实际引用以及检索/生成/引用阶段耗时。自动指标分别检查：事实正则、拒答正则、证据是否在检索结果/最终模型上下文、引用编号合法性、引用是否指向预期证据。引用格式合法、字符串命中或相似度补引均不能证明事实受支持。

多轮改写本身有随机性，原端到端 prompt 对照的证据上下文未完全固定，不能把所有差值归因于提示词。`replay.py` 可重放已捕获的实际模型输入，冻结知识块/引用指令/会话和采样参数，仅替换指令前缀；它是对已见夹具的诊断复验，不是新的留出成绩。

`human-review.csv` 保留 correctness/faithfulness/citationSupport 栏，默认 pending。人工语义核验完成前，不报告“忠实度 xx%”。候选提示词若出现幻觉/错误率/拒答回退，不能仅凭数字题提升通过生产回答门禁；默认产品提示词不自动切换。

### 证据摘录复测

评测 Compose 的证据摘录默认使用 `EVIDENCE_RAG_CHAT_MODEL`，也可配置 `EVIDENCE_RAG_EXTRACTIVE_MODEL` 指定同一个服务的其他模型。

```bash
docker compose -f compose.yaml -f evaluation/compose.eval.yaml up -d --build --wait
python3 evaluation/qa.py --candidate extractive --split dev --url http://127.0.0.1:18081 \
  --out evaluation/runs/answer-v2/qa-dev
python3 evaluation/qa.py --candidate extractive --split test --url http://127.0.0.1:18081 \
  --selection evaluation/runs/answer-v2/qa-dev/selection.json --out evaluation/runs/answer-v2/qa-test
python3 evaluation/squad_qa.py --url http://127.0.0.1:18081
```

中文测试集沿用曾经测过的 18 题，适合判断是否修复已知回归。`--candidate extractive` 运行新版，原提示词与上一版证据优先提示词的对照可从历史报告读取，或用默认参数另跑。脚本将新版 corpus 按文档哈希写入独立数据集，并记录检索时真正见到的片段和最终引用。新版门禁检查有答案题的事实与预期证据同时命中率至少 90%，无答案题全部明确拒答，所有已作答的原文位置校验通过，以及零接口错误。`automatedPassed` 记录自动验收，独立人工复核仍由 `--review` 提交。

公开检验从 SQuAD 2.0 的公开开发集固定抽取 16 个不同文章段落，每段一题有答案、一题无答案，使用原始标注和 SHA256。问题用 `docId` 限定指定段落，分别统计选中证据包含任一金标答案及拒答的比例。这里衡量给定段落的证据选择，正式检索评测使用 SciFact。来源见 [SQuAD 2.0 数据页](https://rajpurkar.github.io/SQuAD-explorer/) 和 [数据集论文](https://aclanthology.org/P18-2124/)。

汇总新版验收时，把 `--retrieval` 和 `--load` 指向本次检索门禁、压测摘要，`--qa` 和 `--qa-selection` 指向本次中文测试摘要、开发集冻结结果。`--out` 写入单独的报告目录，便于保留前后两版记录。

## 工程边界

已具备：批量向量化、批量 ES 写入和目录原子落盘、失败补偿、默认关闭的有界 TTL query cache、模型身份隔离、请求 ID、2 MiB 请求体限制、可选 Bearer token、有界准入与 429/Retry-After、上游 HTTP 超时、embedding 有界重试、数据/配置/代码指纹、完整性校验、门禁退出码、逐题和 bad case 产物。

`EVIDENCE_RAG_MAX_INFLIGHT` 默认为 8（1..256）；`EVIDENCE_RAG_HTTP_TOKEN` 非空时保护业务 API，health 保持可读。HTTP token 是服务级访问控制，不是多租户 ACL。`EVIDENCE_RAG_QUERY_CACHE_ENTRIES` 默认为 0；评测覆盖配置设为 1024，TTL 30 分钟，同一模型实例内有效，异常不缓存。原版默认处理顺序保持不变；新增策略需显式选择。

尚待生产验收：人工标注和双人复核的真实业务集、租户鉴权与证据权限、分布式目录/写锁、持久化作业和崩溃恢复演练、端到端取消/总截止时间、稳定的模型错误恢复、跨节点容量和长时间压力、备份与恢复、生产 SLO/告警、依赖供应链/漏洞检查。当前 CorpusService 仍以 synchronized 串行编排；有限并发测试会明确暴露此瓶颈，不能把有虚拟线程写成已经支持高吞吐。

## 文件

- `data/scifact/manifest.json`：数据来源和冻结版本；不把下载数据纳入源码仓库。
- `runs/*/run.json`：实际配置、源码指纹、运行依赖信息。
- `results.jsonl`：每题排名、得分、延迟、错误、中间步骤；回答结果额外包含模型边界。
- `summary.json/csv`：聚合及配对比较；`cases.jsonl` 包含改善和退化案例。
- `selection.json`：开发集冻结的候选，带数据/配置/代码指纹。
- `gate.json`：仅针对检索的门禁，不对生成/安全/容量作认证。
- `human-review.csv`：回答语义复核待办，未填内容不能记为通过。

测评实现参考 [BEIR 指标定义](https://github.com/beir-cellar/beir/wiki/Metrics-available)。采用本项目固定分母实现，而非直接依赖外部工具的缺失 query 行为。

人工复核完成后可向 `release_gate.py` 传入 `--review path/to/human-review.csv`；它验证选中候选每题的问题/回答一致、覆盖完整、reviewer 非空和二元评分。无答案题的 citationSupport 可填 NA。未传入复核表时保持阻断。
