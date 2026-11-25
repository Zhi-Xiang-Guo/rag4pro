#!/usr/bin/env python3
"""Check the same payment runbook endpoint used by the interactive demo."""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import eval as artifacts
from qa_checks import explicit_refusal, source_text, verified_spans

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT.parent / "src/main/resources/demo/payment.json"
LABELS = ROOT / "payment-qa.json"


def matches(item, chunk):
    return (
        item["docId"] == chunk.get("document_external_id")
        and item["text"] in source_text(chunk)
    )


def evaluate(case, response, documents):
    expected = case["evidence"]
    reference = response.get("reference", {})
    cited_chunks = reference.get("chunks", [])
    retrieved = response.get("evaluation", {}).get("retrievedChunks", [])
    spans = verified_spans(response)
    answers = [
        {"docId": cited_chunks[span["citationIndex"]]["document_external_id"], "text": span["text"]}
        for span in spans
    ]
    in_scope = all(
        documents.get(chunk.get("document_external_id"), {}).get("version") == case["version"]
        for chunk in [*retrieved, *cited_chunks]
    )
    exact_source = bool(answers) and all(
        quote["text"] in documents.get(quote["docId"], {}).get("text", "") for quote in answers
    )
    complete = all(
        any(quote["docId"] == item["docId"] and item["text"] in quote["text"] for quote in answers)
        for item in expected
    )
    relevant = all(
        any(quote["docId"] == item["docId"] and item["text"] in quote["text"] for item in expected)
        for quote in answers
    )
    return {
        "answerStatus": response.get("answerStatus"),
        "answer": response.get("answer", ""),
        "retrievedDocIds": [chunk.get("document_external_id") for chunk in retrieved],
        "quotes": answers,
        "scopePass": in_scope,
        "sourcePass": exact_source if response.get("answerStatus") == "answered" else explicit_refusal(response),
        "evidenceRetrieved": bool(expected) and all(any(matches(item, chunk) for chunk in retrieved) for item in expected),
        "goldCited": bool(expected) and exact_source and complete,
        "onlyGoldCited": bool(expected) and exact_source and complete and relevant,
        "refusalPass": explicit_refusal(response),
        "retrievalMs": response.get("evaluation", {}).get("retrievalMs"),
    }


def summarize(rows):
    answerable = [row for row in rows if row["evidence"]]
    unanswerable = [row for row in rows if not row["evidence"]]
    count = lambda key, items: sum(bool(row.get(key)) for row in items)
    summary = {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "goldEvidenceRetrieved": count("evidenceRetrieved", answerable),
        "goldEvidenceCited": count("goldCited", answerable),
        "onlyGoldEvidenceCited": count("onlyGoldCited", answerable),
        "correctRefusals": count("refusalPass", unanswerable),
        "citationAndRefusalIntegrity": count("sourcePass", rows),
        "scopePass": count("scopePass", rows),
        "errors": count("error", rows),
        "latencyP95Ms": round(artifacts.percentile([row["latencyMs"] for row in rows], 0.95), 1),
    }
    summary["passed"] = (
        bool(answerable) and bool(unanswerable)
        and summary["errors"] == 0
        and summary["scopePass"] == len(rows)
        and summary["citationAndRefusalIntegrity"] == len(rows)
        and summary["goldEvidenceCited"] / len(answerable) >= 0.90
        and summary["correctRefusals"] / len(unanswerable) >= 0.90
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--dev-summary", type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.dev_summary:
        parser.error("--split test requires --dev-summary from the frozen dev run")
    if (args.out / "summary.json").exists() or (args.out / "results.jsonl").exists():
        parser.error("choose a fresh --out directory")

    fixture_bytes = FIXTURE.read_bytes()
    documents = {doc["docId"]: doc for doc in json.loads(fixture_bytes)["documents"]}
    labels_bytes = args.labels.read_bytes()
    all_cases = json.loads(labels_bytes)["queries"]
    cases = [case for case in all_cases if case["split"] == args.split]
    if not cases or len({case["id"] for case in all_cases}) != len(all_cases):
        raise ValueError("empty split or duplicate query ID")
    for case in cases:
        for item in case["evidence"]:
            source = documents[item["docId"]]
            if source["version"] != case["version"] or item["text"] not in source["text"]:
                raise ValueError(f"invalid gold evidence: {case['id']}")

    health = artifacts.request(args.url, "/api/health")
    profile = {
        "fixtureSha256": artifacts.digest(fixture_bytes),
        "labelsSha256": artifacts.digest(labels_bytes),
        "sourceSha256": artifacts.source_fingerprint(),
        "checkerSha256": artifacts.digest(Path(__file__).read_bytes() + (ROOT / "qa_checks.py").read_bytes()),
        "embeddingIdentitySha256": artifacts.digest(health["embeddingIdentity"].encode()),
        "evidenceModel": health["extractiveModel"],
        "context": health["extractiveContext"],
    }
    if args.dev_summary:
        dev = artifacts.read(args.dev_summary)
        if dev["split"] != "dev" or dev["profile"] != profile:
            raise ValueError("fixture, source, checker, or runtime changed after the dev run")

    scenario = artifacts.request(args.url, "/api/demo", method="POST")
    if scenario["revision"] != profile["fixtureSha256"] or scenario["indexedCount"] != len(documents):
        raise ValueError("server fixture differs or corpus is incomplete")
    for doc_id, document in documents.items():
        route = "/api/demo/source?" + urlencode({"docId": doc_id})
        if artifacts.request(args.url, route) != document:
            raise ValueError(f"original source differs: {doc_id}")

    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.jsonl").open("x", encoding="utf-8") as output:
        for case in cases:
            started = time.monotonic()
            try:
                request = {key: case[key] for key in ("version", "question")}
                response = artifacts.request(args.url, "/api/demo/chat", request, timeout=100)
                result = evaluate(case, response, documents)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                result = {"error": str(error)[:500]}
            row = {**case, **result, "latencyMs": round((time.monotonic() - started) * 1000, 1)}
            rows.append(row)
            output.write(artifacts.canonical(row).decode() + "\n")
            output.flush()
            passed = row.get("goldCited") if case["evidence"] else row.get("refusalPass")
            print(f"{case['id']} {'PASS' if passed else 'FAIL'}", flush=True)

    summary = {"split": args.split, "profile": profile, **summarize(rows)}
    artifacts.write(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
