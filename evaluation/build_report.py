#!/usr/bin/env python3
"""Export a self-contained, traceable interview evidence pack from completed runs."""
import argparse,datetime,gzip,hashlib,json,pathlib,shutil
import eval as e

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default=str(e.ROOT/'runs'));p.add_argument('--out',default=str(e.ROOT/'reports/2026-11-24'));a=p.parse_args()
    runs=pathlib.Path(a.runs);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    test=e.read(runs/'test/summary.json');dev=e.read(runs/'dev/summary.json');meta=e.read(runs/'test/run.json');qa=e.read(runs/'qa-test/summary.json');gate=e.read(runs/'release-gate.json')
    selected=meta['selection']['selected'];baseline=meta['config']['baseline'];b=test['variants'][baseline];c=test['variants'][selected];comparison=test['comparisons'][selected]
    if not test['complete'] or not qa['complete']:raise ValueError('cannot package incomplete experiment')
    for name in ('dev','test','qa-dev','qa-test','load','replay'):
        for src in (runs/name).glob('*'):
            if not src.is_file():continue
            target=out/name/src.name;target.parent.mkdir(parents=True,exist_ok=True)
            if src.suffix=='.jsonl':
                with open(src,'rb') as f,gzip.GzipFile(filename=str(target)+'.gz',mode='wb',mtime=0) as g:shutil.copyfileobj(f,g)
            else:shutil.copy2(src,target)
    if (runs/'implementation.json').exists():shutil.copy2(runs/'implementation.json',out/'implementation.json')
    shutil.copy2(runs/'release-gate.json',out/'release-gate.json')
    shutil.copy2(e.ROOT/'data/scifact/manifest.json',out/'dataset-manifest.json')
    shutil.copy2(e.ROOT/'qa-fixture.json',out/'qa-fixture.json')
    for src in (e.ROOT/'configs').glob('*.json'):shutil.copy2(src,out/src.name)
    (out/'DATA-LICENSE.md').write_text('''# Dataset attribution\n\nScientific document excerpts and labels in the evidence artifacts originate from BEIR SciFact, distributed under CC-BY-SA-4.0 according to its dataset card: https://huggingface.co/datasets/BeIR/scifact . Revisions, hashes and source URLs are recorded in dataset-manifest.json.\n\nSciFact: David Wadden et al., Fact or Fiction: Verifying Scientific Claims, EMNLP 2020 (https://aclanthology.org/2020.emnlp-main.609/). BEIR: Nandan Thakur et al., BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models (https://openreview.net/forum?id=wCu6T5xFjeJ).\n\nControlled QA fixture is developer-authored synthetic data, not company policy. No independent human review has been completed.\n''')
    pct=lambda x:f'{100*x:.2f}%'
    delta=lambda key:f'{100*(c[key]-b[key]):+.2f} 个百分点'
    rel=(c['ndcg@10']/b['ndcg@10']-1)*100 if b['ndcg@10'] else None
    ci=comparison['pairedBootstrap95CI']
    lines=['# RAGFlow4j 测评与面试证据（2026-11-24）','',
        f'完整 SciFact 5,183 篇语料；开发集 60 题，官方测试集 300 题。开发集按 nDCG@10 选中 `{selected}`，测试阶段比较 `{baseline}` 与该冻结候选。', '',
        '| 留出测试指标 | Java 原版默认路径 | 开发集选中的候选 | 差值 |','|---|---:|---:|---:|']
    for key in ('ndcg@10','recall@10','mrr@10'):
        lines.append(f'| {key} | {pct(b[key])} | {pct(c[key])} | {delta(key)} |')
    lines += ['',f'nDCG@10 相对变化 {rel:+.2f}%；绝对变化配对 bootstrap 95% 区间 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] 个百分点。胜/平/负为 {comparison["wins"]}/{comparison["ties"]}/{comparison["losses"]}，两组错误数 {b["errors"]}/{c["errors"]}。相近断言的相关性可能使按 query bootstrap 区间偏窄。', '',
        '## 可以用于简历的表述','',
        f'基于 Java 21 与 Elasticsearch 移植并梳理 RAGFlow 标准文本链路，搭建包含数据版本、分层指标、开发/测试隔离、逐题追踪、失败记零和质量门禁的评测流程；在完整 SciFact 语料与 300 条官方测试问题上，对原版 Java 默认检索路径与独立召回策略进行对照，选中 `{selected}`，nDCG@10 从 {pct(b["ndcg@10"])} 变化到 {pct(c["ndcg@10"])}，Recall@10 从 {pct(b["recall@10"])} 变化到 {pct(c["recall@10"])}，保留退化案例和置信区间。', '',
        '这是个人项目和公开集实验；不要写成公司生产业务收益，也不要称为官方 RAGFlow 与全量 Java 端口的胜负。只有实际提升的指标可以使用“提升”措辞。', '',
        '## 两分钟讲述','',
        '1. 问题：只演示几道题无法证明 RAG 改动有效。我先拆分索引、召回、排序、上下文、生成和引用的责任边界。',
        '2. 基线：保留原 Java 移植默认路径。发现 mode=bm25 仍走 embedding/kNN，所以另建真正独立词法路径，避免错误消融。',
        '3. 假设：向量召回受词法前置过滤可能漏掉语义相关文本。增加独立 dense 与两路 RRF，对齐总候选预算 64，并分别记录计算成本。',
        f'4. 实验：同一 embedding、同一索引和数据快照；60 题 dev 只负责选参，300 题 test 只负责验证。候选为 `{selected}`。文档级指标先去重 chunk，失败保留分母。',
        f'5. 结果：nDCG@10 {delta("ndcg@10")}、Recall@10 {delta("recall@10")}。通过逐题排名、胜平负、回归样例和配对区间核验结果。',
        '6. 边界：科学断言检索成绩不能代表中文业务问答正确率。生成另用本地小模型跑合成回归，暴露幻觉与拒答问题，整体上线门禁保持阻断。', '',
        '## 回答和引用的实测','',
        '下表是 18 条留出合成问题（15 条有答案、3 条无答案）的规则检查，尚未独立人工核验。原提示词与候选提示词均运行相同本地 qwen2.5:0.5b。', '',
        '| 指标 | 原提示词 | 证据优先提示词 |','|---|---:|---:|']
    qb=qa['variants']['original'];qc=qa['variants'].get('evidence_first',qb)
    for key,label in [('factPatternPassRate','数字/单位事实正则通过率'),('refusalPatternPassRate','无答案拒答正则通过率'),('evidenceRecall','检索预期证据命中率'),('finalContextEvidenceRate','最终模型上下文证据率'),('expectedEvidenceCitedRate','引用指向预期证据率')]:
        lines.append(f'| {label} | {pct(qb[key])} | {pct(qc[key])} |')
    lines += [f'| 上游错误数 | {qb["errors"]} | {qc["errors"]} |','',
        '不能把正则命中说成语义准确率，不能把引用合法说成事实忠实。多轮 query refiner 具有随机性，两组上下文并未完全相同，所以差值不能全部归因于提示词。另有固定上下文的诊断重放。候选提示词在无答案问题上出现编造人名，另有本地模型 HTTP 500；原始失败完整保留。', '',
        f'整体评测验收门禁：**{"PASS" if gate["passed"] else "BLOCKED"}**。阻断原因：', '']
    lines += ['- '+failure for failure in gate['failures']]
    load=e.read(runs/'load/summary.json')
    lines += ['', '## 延迟与有限并发','',
        '检索延迟测于本机 Docker、每题向量显式预热、组别顺序打散；不包含冷 embedding 网络成本。',
        f'测试集原路径 P50/P95 = {b["latencyP50Ms"]:.1f}/{b["latencyP95Ms"]:.1f} ms；候选 = {c["latencyP50Ms"]:.1f}/{c["latencyP95Ms"]:.1f} ms。这是观测结果，不是生产 SLO。standard 返回向量与详细 DSL，新策略响应更小，差异包括评分、序列化和传输成本。', '',
        '| 并发 | 成功/总请求 | 429 | 其他错误 | 成功 P95 ms | 成功 RPS |','|---|---:|---:|---:|---:|---:|']
    for r in load['runs']:lines.append(f'| {r["concurrency"]} | {r["successes"]}/{r["requests"]} | {r["429"]} | {r["otherFailures"]} | {r["successP95Ms"]:.1f} | {r["successfulRps"]:.2f} |')
    diagnostic=e.read(runs/'test/candidate-diagnosis.json') if (runs/'test/candidate-diagnosis.json').exists() else None
    if diagnostic:
        db=diagnostic['variants']['standard'];dc=diagnostic['variants']['rrf']
        lines += ['', '## 三个可以讲清楚的真实案例','',
            '- Query 660（伊维菌素与盘尾丝虫病）：金标文献 1215116 在原路径最终排名第 26，RRF 排名第 1；独立词法分支第 1、独立 dense 第 4。这个案例改善来自排序，不能描述为原路径完全没召回。',
            '- Query 870（肥胖与生活质量）：金标文献 195689316 未进入原路径返回的 64 候选，独立 dense 第 3，RRF 第 9；词法分支前 64 未出现它。这条展示独立向量分支对候选覆盖的帮助。',
            '- Query 1088（Bcl2 与肿瘤）：金标文献 37549932 从原路径第 1 退到 RRF 第 6。它是 dense 分支第 1，却未进入词法前 64；RRF 对两路共同出现的其他候选更有利，暴露融合的共识偏置。',
            f'事后候选诊断：原路径平均返回 {db["meanReturnedDocuments"]:.2f} 篇文档、全返回集合 Recall {pct(db["allReturnedDocumentRecall"])}；RRF 平均 {dc["meanReturnedDocuments"]:.2f} 篇、Recall {pct(dc["allReturnedDocumentRecall"])}。这些诊断未参与选参；Top-10 的提升同时含候选覆盖和重排序影响。',
            '案例的独立分支排名来自测试后诊断请求，不是重新调参；不能据此把组合改动的全部收益归因于单个机制。']
    if (runs/'replay/summary.json').exists():
        replay=e.read(runs/'replay/summary.json');rb=replay['variants']['original'];rc=replay['variants']['evidence_first']
        lines += ['', '## 固定上下文的生成诊断复验','',
            '对已见的 18 条合成 test 夹具重放原始最终 messages，冻结知识块/引用规则/会话/seed，单独替换提示指令前缀；不再运行检索与多轮改写。这是诊断复验，不能当作新的留出泛化成绩。',
            f'原指令/候选指令的事实正则通过率为 {pct(rb["factPatternPassRate"])}/{pct(rc["factPatternPassRate"])}；无答案拒答正则通过率为 {pct(rb["refusalPatternPassRate"])}/{pct(rc["refusalPatternPassRate"])}。错误数 {rb["errors"]}/{rc["errors"]}。固定证据后仍出现拒答退化，因此该提示词不自动上线。',
            '完整输入及原始输出保存在 replay/results.jsonl.gz；知识与引用规则尾部具有相同的 SHA256。']
    lines += ['', '## 追问准备','',
        '- 为什么不用纯向量分数与词法分数直接相加？量纲、分布与校准不同；RRF 用名次融合。也通过独立 dense 对照验证融合是否真的必要。',
        '- 为什么不用每个 chunk 当作命中？同一篇文档可能切出多个块，重复计数会虚增指标；文档级与证据级分别统计。',
        '- 为什么结果可以复查？数据 commit/SHA256、配置和源码指纹、镜像 ID、逐题排名、失败、选择记录、压缩原始记录都在证据包。',
        '- 为什么原版仍保留？用于学习和回归基线，新增策略显式 opt-in；不能把改进路径称为“与原版完全一致”。',
        '- 哪里还没达到生产要求？真实业务人工金标、租户 ACL、持久化作业/分布式目录、总截止时间与取消、模型恢复、长期压测与备份恢复尚需验收。',
        '- 最大的负结果是什么？小模型可以引用真实片段却编造答案；因此检索、事实、引用编号和引用支持需要分别检查。', '',
        '## 复查产物','',
        '`test/results.jsonl.gz` 为全部逐题排名，`test/cases.jsonl.gz` 同时包含改善和退化；`qa-test/results.jsonl.gz` 包含实际最终 messages、原回答和引用；`human-review.csv` 仍待人工复核。`SHA256SUMS.json` 固定本包内容。',
        '', '来源：[BEIR SciFact 数据卡](https://huggingface.co/datasets/BeIR/scifact)、[BEIR 数据列表](https://github.com/beir-cellar/beir/wiki/Datasets-available)、[RRF 原论文](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf)。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Preserve the exact implementation for reviewing this evidence without copying credentials/corpus downloads.
    for src in list((e.ROOT.parent/'src/main/java').rglob('*.java'))+list(e.ROOT.glob('*.py'))+[e.ROOT.parent/'pom.xml',e.ROOT.parent/'LICENSE',e.ROOT.parent/'NOTICE']:
        target=out/'source'/src.relative_to(e.ROOT.parent);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    manifest={str(f.relative_to(out)):e.digest(f.read_bytes()) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SHA256SUMS.json'}
    e.write(out/'SHA256SUMS.json',manifest)
    print(out/'REPORT.md')
if __name__=='__main__':main()
