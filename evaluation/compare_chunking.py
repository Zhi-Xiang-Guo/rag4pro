"""Paired chunking and retrieval ablations on HotpotQA's distractor validation split."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
from urllib.parse import urlencode

import eval as evaluation

SEED = "rag4pro-hotpot-context-v1"
REQUEST = {
    "topK": 8, "candidateK": 32, "knnTopK": 32, "knnNumCandidates": 128,
    "evidenceBudget": 512, "language": "English", "aggs": False
}
CHUNK_TOKENS = 64
VARIANTS = {
    "fixed_rrf_leaf": ("fixed", "rrf", "leaf"),
    "hierarchical_rrf_leaf": ("hierarchical", "rrf", "leaf"),
    "hierarchical_rrf_parent": ("hierarchical", "rrf", "parent"),
    "hierarchical_rrf_section": ("hierarchical", "rrf", "section"),
    "hierarchical_diverse_section": ("hierarchical", "rrf_diverse", "section")
}


def read_questions(source, split):
    import pyarrow.parquet as parquet

    rows = parquet.read_table(source).to_pylist()
    if len(rows) != 7405 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected the 7,405-row HotpotQA distractor validation split")
    order = sorted(rows, key=lambda row: evaluation.digest(f"{SEED}/{row['id']}".encode()))
    return order[:24] if split == "calibration" else order[24:124]


def prepare(rows):
    documents, questions = {}, []
    for row in rows:
        by_title = {}
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            text = " ".join(sentence.strip() for sentence in sentences)
            doc_id = evaluation.digest(evaluation.canonical([title, sentences]))[:32]
            if title in by_title or not text.strip():
                raise ValueError(f"Ambiguous context for {row['id']}")
            by_title[title] = (doc_id, sentences)
            documents[doc_id] = {"docId": doc_id, "title": title, "text": text, "fileType": "txt",
                                 "language": "English", "chunkTokenSize": CHUNK_TOKENS}
        if not 2 <= len(by_title) <= 10:
            raise ValueError(f"Unexpected number of provided candidates for {row['id']}")
        facts = []
        for title, sentence in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
            doc_id, sentences = by_title[title]
            facts.append((doc_id, sentence, sentences[sentence]))
        if not facts:
            raise ValueError(f"Missing supporting facts for {row['id']}")
        questions.append({"id": row["id"], "question": row["question"],
                          "docIds": [doc_id for doc_id, _ in by_title.values()],
                          "facts": list(dict.fromkeys(facts))})
    return documents, questions


def index_documents(url, dataset_id, strategy, documents, corpus_sha, embedding):
    path = "/api/documents?" + urlencode({"datasetId": dataset_id})
    existing = {row["docId"]: row for row in evaluation.request(url, path)}
    if existing.keys() - documents.keys():
        raise ValueError(f"Unexpected documents in {dataset_id}")
    for doc_id, entry in existing.items():
        if (entry.get("metadata", {}).get("corpusSha256") != corpus_sha
                or entry.get("sourceSha256") != evaluation.digest(documents[doc_id]["text"].encode())
                or entry.get("embeddingIdentity") != embedding + "|ragflow-title-blend-v1"
                or entry.get("chunkStrategy") != strategy or entry.get("chunkTokenSize") != CHUNK_TOKENS):
            raise ValueError(f"Index provenance mismatch: {dataset_id}/{doc_id}")
    pending = [doc for doc_id, doc in documents.items() if doc_id not in existing]
    for start in range(0, len(pending), 40):
        batch = [dict(doc, chunkStrategy=strategy, metadata={"corpusSha256": corpus_sha})
                 for doc in pending[start:start + 40]]
        result = evaluation.request(url, "/api/documents/batch",
                                    {"datasetId": dataset_id, "documents": batch}, timeout=900)
        if set(result["docIds"]) != {doc["docId"] for doc in batch}:
            raise ValueError("Ingest returned the wrong document identities")
        print(f"{dataset_id}: {len(existing) + start + len(batch)}/{len(documents)}", flush=True)
    final = evaluation.request(url, path)
    if len(final) != len(documents) or {row["docId"] for row in final} != documents.keys():
        raise ValueError("Incomplete corpus")
    return sum(row["chunkCount"] for row in final)


def normalize(text):
    return " ".join(text.split())


def score(question, result):
    if result["evidenceTokens"] > REQUEST["evidenceBudget"]:
        raise ValueError("Evidence exceeded the configured budget")
    ranked = list(dict.fromkeys(hit["document_external_id"] for hit in result["hits"]))
    if set(ranked) - set(question["docIds"]):
        raise ValueError("Retrieval escaped the question's candidate scope")
    passages = defaultdict(list)
    for chunk in result["evidence"]:
        doc_id = chunk["document_external_id"]
        if doc_id not in question["docIds"]:
            raise ValueError("Evidence escaped the question's candidate scope")
        passages[doc_id].append(normalize(chunk["content"]))
    gold = question["facts"]
    gold_docs = {doc_id for doc_id, _, _ in gold}
    found = [(doc_id, pos) for doc_id, pos, text in gold
             if any(normalize(text) in passage for passage in passages[doc_id])]
    return {"ranking": ranked, "foundFacts": found, "goldFacts": [(doc_id, pos) for doc_id, pos, _ in gold],
            "supportDocumentRecall": len(gold_docs.intersection(ranked)) / len(gold_docs),
            "supportSentenceRecall": len(found) / len(gold), "allSupportSentences": len(found) == len(gold),
            "evidenceTokens": result["evidenceTokens"], "retrievalMs": result["retrievalMs"]}


def report(out, rows, expected):
    summary = {}
    for variant in VARIANTS:
        paired = [row for row in rows if row["variant"] == variant]
        if len(paired) != expected or len({row["id"] for row in paired}) != expected:
            raise ValueError(f"Incomplete run: {variant}")
        summary[variant] = {
            "questions": expected,
            "supportDocumentRecall": statistics.mean(row["supportDocumentRecall"] for row in paired),
            "supportSentenceRecall": statistics.mean(row["supportSentenceRecall"] for row in paired),
            "allSupportSentences": sum(row["allSupportSentences"] for row in paired),
            "meanEvidenceTokens": statistics.mean(row["evidenceTokens"] for row in paired),
            "p95RetrievalMs": evaluation.percentile([row["retrievalMs"] for row in paired], .95)
        }
    evaluation.write(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18082")
    parser.add_argument("--split", choices=("calibration", "holdout"), required=True)
    parser.add_argument("--source", type=Path, default=evaluation.ROOT / "data/public-2026-11-26/hotpot-dev-distractor.parquet")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    documents, questions = prepare(read_questions(args.source, args.split))
    corpus_sha = evaluation.digest(evaluation.canonical(documents))
    health = evaluation.request(args.url, "/api/health")
    suffix = evaluation.digest(evaluation.canonical([corpus_sha, CHUNK_TOKENS, health["embeddingIdentity"]]))[:14]
    datasets = {strategy: f"hotpot-{suffix}-{strategy}" for strategy in ("fixed", "hierarchical")}
    args.out.mkdir(parents=True, exist_ok=True)
    profile = {
        "dataset": "HotpotQA distractor validation", "split": args.split, "seed": SEED,
        "parquetSha256": evaluation.digest(args.source.read_bytes()), "corpusSha256": corpus_sha,
        "questions": [q["id"] for q in questions], "documents": len(documents), "datasets": datasets,
        "embedding": health["embeddingIdentity"], "queryCacheEntries": health["queryCacheEntries"],
        "chunkTokens": CHUNK_TOKENS, "request": REQUEST, "variants": {k: list(v) for k, v in VARIANTS.items()},
        "sourceSha256": evaluation.source_fingerprint(),
        "evaluationSha256": evaluation.digest(Path(__file__).read_bytes())
    }
    run = args.out / "run.json"
    if run.exists() and evaluation.read(run) != profile:
        raise ValueError("The run's source, data or configuration changed")
    evaluation.write(run, profile)
    chunks = {strategy: index_documents(args.url, ds, strategy, documents, corpus_sha, health["embeddingIdentity"])
              for strategy, ds in datasets.items()}
    evaluation.write(args.out / "index-summary.json", chunks)
    path = args.out / "results.jsonl"
    rows = evaluation.rows(path) if path.exists() else []
    done = {(row["id"], row["variant"]) for row in rows}
    if len(done) != len(rows):
        raise ValueError("Duplicate results")
    for index, question in enumerate(questions):
        body = dict(REQUEST, question=question["question"], docIds=question["docIds"])
        plan = list(VARIANTS)
        offset = index % len(plan)
        todo = [variant for variant in plan[offset:] + plan[:offset] if (question["id"], variant) not in done]
        if todo:
            evaluation.request(args.url, "/api/retrieve/context",
                               dict(body, strategy="dense", contextMode="leaf", datasetId=datasets["fixed"], topK=1))
        for variant in todo:
            strategy, retrieval, mode = VARIANTS[variant]
            started = time.monotonic()
            result = evaluation.request(args.url, "/api/retrieve/context",
                                        dict(body, datasetId=datasets[strategy], strategy=retrieval, contextMode=mode))
            scored = score(question, result)
            record = {"id": question["id"], "variant": variant, "requestMs": (time.monotonic() - started) * 1000,
                      **scored}
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append(record)
        print(f"{args.split}: {index + 1}/{len(questions)}", flush=True)
    report(args.out, rows, len(questions))


if __name__ == "__main__":
    main()
