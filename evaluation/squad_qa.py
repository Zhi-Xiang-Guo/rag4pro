#!/usr/bin/env python3
"""Document-scoped QA and refusal challenge on a fixed sample of public SQuAD 2.0 dev."""

import argparse
import json
import pathlib
import re
import time

import eval as e
from qa_checks import explicit_refusal, verified_spans


SOURCE = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json"
SOURCE_SHA = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
SEED = "ragflow4j-squad2-challenge-v1"
ARTICLES = 16


def order(value):
    return e.digest(f"{SEED}/{value}".encode())


def sample(raw):
    documents, queries = [], []
    for article in sorted(raw["data"], key=lambda item: order(item["title"])):
        for paragraph in sorted(article["paragraphs"], key=lambda item: order(item["context"])):
            question_rows = sorted(paragraph["qas"], key=lambda item: order(item["id"]))
            positive = next((q for q in question_rows if not q["is_impossible"] and q["answers"]), None)
            negative = next((q for q in question_rows if q["is_impossible"]), None)
            if positive is None or negative is None:
                continue
            context = paragraph["context"]
            doc_id = order(context)[:16]
            documents.append({"docId": doc_id, "title": article["title"].replace("_", " "), "language": "English",
                              "fileType": "txt", "text": context, "metadata": {"squad_sha256": SOURCE_SHA}})
            for row in (positive, negative):
                answers = {a["text"] for a in row["answers"]
                           if context[a["answer_start"]:a["answer_start"] + len(a["text"])] == a["text"]}
                if not row["is_impossible"] and not answers:
                    raise ValueError(f"SQuAD label offsets do not match context for {row['id']}")
                queries.append({"id": row["id"], "docId": doc_id, "question": row["question"],
                                "unanswerable": row["is_impossible"], "acceptableSpans": sorted(answers)})
            break
        if len(documents) == ARTICLES:
            break
    if len(documents) != ARTICLES or len(queries) != 2 * ARTICLES:
        raise ValueError("Missing challenge articles")
    return documents, sorted(queries, key=lambda item: order(item["id"]))


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def has_label(spans, labels):
    return any(f" {normalize(label)} " in f" {normalize(span['text'])} "
               for span in spans for label in labels if normalize(label))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18081")
    parser.add_argument("--data", default=str(e.ROOT / "data/squad-v2/dev-v2.0.json"), type=pathlib.Path)
    parser.add_argument("--out", default=str(e.ROOT / "runs/answer-v2/squad"), type=pathlib.Path)
    args = parser.parse_args()
    raw = args.data.read_bytes() if args.data.exists() else e.fetch(SOURCE)
    if e.digest(raw) != SOURCE_SHA:
        raise ValueError("SQuAD 2.0 checksum mismatch")
    if not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        args.data.write_bytes(raw)
    documents, queries = sample(json.loads(raw))
    dataset = "eval-squad2-scoped-v1"
    health = e.request(args.url, "/api/health")
    meta = {"dataset": dataset, "sourceUrl": SOURCE, "sourceSha256": SOURCE_SHA, "seed": SEED,
            "paper": "https://aclanthology.org/P18-2124/", "license": "CC-BY-SA-4.0",
            "scope": "16 distinct public dev articles, one positive and one hard negative per paragraph, doc-scoped",
            "sampleSha256": e.digest(e.canonical({"documents": documents, "queries": queries})),
            "sourceFingerprint": e.source_fingerprint(), "runnerSha256": e.digest(pathlib.Path(__file__).read_bytes()),
            "runtime": {key: health[key] for key in ("embeddingIdentity", "extractiveModel", "extractiveContext")}}
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "run.json").exists() and e.read(args.out / "run.json") != meta:
        raise ValueError("SQuAD challenge fingerprint drift, use a fresh output directory")
    e.write(args.out / "run.json", meta)
    known = {doc["docId"]: doc for doc in e.request(args.url, f"/api/documents?datasetId={dataset}")}
    if any(doc.get("metadata", {}).get("squad_sha256") != SOURCE_SHA for doc in known.values()):
        raise ValueError("Indexed SQuAD source fingerprint drift")
    pending = [doc for doc in documents if doc["docId"] not in known]
    if pending:
        e.request(args.url, "/api/documents/batch", {"datasetId": dataset, "documents": pending})
    result_file = args.out / "results.jsonl"
    records = e.rows(result_file) if result_file.exists() else []
    completed = {row["id"] for row in records}
    for query in queries:
        if query["id"] in completed:
            continue
        body = {"answerMode": "extractive", "strategy": "rrf", "datasetId": dataset, "docId": query["docId"],
                "question": query["question"], "topK": 4, "evaluationTrace": True}
        start = time.monotonic()
        error = None
        response = {}
        try:
            response = e.request(args.url, "/api/chat", body, timeout=90)
        except Exception as exc:
            error = str(exc)[:500]
        spans = verified_spans(response)
        context_presented = response.get("evaluation", {}).get("modelBoundary", {}).get("modelInvoked", False)
        record = dict(query, response=response, latencyMs=(time.monotonic() - start) * 1000, error=error,
                      labelInVerifiedSpan=has_label(spans, query["acceptableSpans"]),
                      contextPresented=context_presented,
                      explicitRefusal=context_presented and explicit_refusal(response), exactSpanVerification=bool(spans))
        with result_file.open("a") as handle:
            handle.write(e.canonical(record).decode() + "\n")
        records.append(record)
        print(f"{len(records)}/{len(queries)}", flush=True)
    if len(records) != len(queries) or {r["id"] for r in records} != {q["id"] for q in queries}:
        raise ValueError("Challenge results incomplete or duplicated")
    positive = [r for r in records if not r["unanswerable"]]
    negative = [r for r in records if r["unanswerable"]]
    answered = [r for r in records if r["response"].get("answerStatus") == "answered"]
    summary = {"questions": len(records), "positive": len(positive), "negative": len(negative),
               "goldInVerifiedSpan": sum(r["labelInVerifiedSpan"] for r in positive),
               "explicitRefusals": sum(r["explicitRefusal"] for r in negative),
               "answered": len(answered), "verifiedAnswers": sum(r["exactSpanVerification"] for r in answered),
               "contextsPresented": sum(bool(r["contextPresented"]) for r in records),
               "errors": sum(r["error"] is not None for r in records),
               "latencyP95Ms": e.percentile([r["latencyMs"] for r in records], .95),
               "scope": "Paragraph-scoped spot check, gold-span presence and refusal. No official SQuAD EM/F1."}
    e.write(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
