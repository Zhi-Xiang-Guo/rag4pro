# 标准链路验证记录

日期为 2026-11-09，时区 Asia/Shanghai。

## 基准和输入

对照本地 RAGFlow 提交 `e4bd0801d5169c3dc0847ce0ea06fc6c3e516105`。使用原 native analyzer 代码和词典资源，ES 版本为原 Compose 默认的 8.11.3。

embedding 使用公司 `qwen3-embedding-4b`，实际返回 2560 维。回答使用本机 Ollama `qwen2.5:0.5b`，Java 容器通过 host.docker.internal:11434 访问。

## 已执行验证

- Java 21 `mvn clean verify`，132 个测试，零失败、零错误、零跳过，包含缓存和 metadata/原生 JSON 修复回归。
- Docker 构建内执行同一测试集，并直接启动成品 JAR 的 `--check-runtime`，验证 native 动态库、cl100k 和 Qwen 资源可用。
- `StandardLiveIT` 实际通过 HTTP 建索引、调用网关 embedding、执行 ES 检索、调用本地模型并返回引用和 usage，1 项测试通过。
- 真实父子块实验产生 6 个子块、2 个隐藏父块。检索返回子块，聊天路径按 mom_id 回填父块。
- 当前示例 `demo-standard/shop-policy` 使用 64-token 目标和零重叠，产生 2 个原始块。

## 独立核对

- native analyzer 原始代码与固定资源版本可追溯，SHA256 manifest 验证通过。
- WordNet synsets 与运行原始 Go 源码的 oracle 比较。
- Qwen BPE 与相同资源的 HuggingFace tokenizer 实际 token ID 比较。
- 检索请求、评分和上下文测试使用原始 Go 来源的条件与预期结果。
- 集成审查修正了数据集语言继承、rankFeature/rerankModel 接线、重复块计数、解析临时字段清理、metadata 范围和删除失败恢复。

## 局限

端到端链路测试没有将本地 0.5B 模型的文本回答判定为事实准确。测试检查相关知识确实进入 prompt、模型实际返回内容、references 与 usage 存在。模型的真实误答已记录在 [coverage.md](alignment/coverage.md)。

所有 RAGFlow 输入与可选分支的严格等价性尚未得到全面证明。具体已验证范围与剩余兼容性范围同样列于 coverage.md。

## 补充验证

正文缓存的 key、未加权向量保存、标题权重变化、TTL、读写故障和任务清理有单独测试。metadata 的 manual/auto/semi_auto、比较运算、原始空匹配语义及损坏 JSON 由同版本 kaptinlin/jsonrepair 辅助程序验证。该辅助程序使用纯 Go 静态编译，部署镜像包含 Linux 可执行文件。

## 最终运行复验

最终 Docker 中 Java 与 Elasticsearch 均为 healthy。发生过一次构建高峰期间的 ES OOM，已恢复原数据卷，限制 Maven/测试 JVM 内存并增加 ES 自动重启，受限构建通过。恢复后的 StandardLiveIT 再次通过。浏览器实际重新建索引得到 3 块，并执行检索与本地回答。本次页面回答文字为“三个工作日”，引用编号由模型生成，仍应核对引用块内容。

## 2026-11-10：代码格式与中文导读

本次为阅读体验整理：统一 Java 主代码与测试格式，增加中文职责说明、检索阶段注释、包说明，以及 retrieve 请求跟读、参数字段表和断点练习。Spotless 与 Google Java Format 版本固定在 `pom.xml`，格式检查接入 `verify`。

- Java 21 执行 `mvn -B spotless:apply verify -DargLine=-Xmx768m`：132 项测试，零失败、零错误、零跳过；53 个 Java 文件通过格式检查。
- 使用 JDK 编译器解析修改前后源码，排除文档注释后比较语法树：52 个原有 Java 文件一致，新增文件仅为 `package-info.java` 包文档。原 Maven 配置除新增 Spotless 外保持一致。
- 执行 `scripts/check-package.sh`：成品 JAR 的 native analyzer、Cl100k 和 Qwen BPE 资源检查通过。
- 只读检查 `docker compose ps` 与 `/api/health`：Java 和 Elasticsearch 均为 healthy，接口返回 `status=ok`。本次没有重建或重启容器，没有重跑调用真实模型的 `StandardLiveIT`。

测试输出中的 Surefire native stream 警告来自原生分词器直接输出词典加载日志；对应 132 项测试仍全部完成。失败降级测试中出现的模型不可用与评分长度警告是测试主动构造的场景。

## 2026-11-10：改为 IDEA 的 ⌘⌥L 格式

按后续要求，使用 IntelliJ IDEA 2026.2.2 自带格式化器与当前 Default 代码风格重新格式化。覆盖 53 个 Java 主代码/测试文件，以及 HTML、XML、JSON、YAML 和脚本，共 67 个 IDEA 支持的文件；原生 Go 辅助程序执行 gofmt，C++ 字符串释放辅助函数整理为多行。第三方上游代码、词典与模型资源保持原样。

本次格式取代上一节的 Google Java Format 规则，已移除此前新增的 Spotless 插件，避免两种风格冲突。IDEA 格式化器 dry-run 确认 67 个文件均符合当前风格；Java 21 的 `mvn verify` 通过 132 项测试，零失败、零错误、零跳过。

53 个 Java 文件在去除文档注释后的语法树与本次操作前一致；JSON 数据及 HTML 表单初始值保持一致，内联 JavaScript 和 shell 语法检查通过。原 Maven 构建配置除移除 Spotless 外保持一致，原生辅助代码仅有空白格式变化。
