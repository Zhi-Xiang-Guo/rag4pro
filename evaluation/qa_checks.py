"""Automatic checks for source spans and explicit refusal in extractive answers."""

import json

REFUSALS = {
    "资料不足，无法回答。",
    "No supporting evidence was found in the knowledge base.",
}


def source_text(chunk):
    return chunk.get("content", chunk.get("content_with_weight", ""))


def verified_spans(response):
    """Recover spans only when the original offsets, citations, and rendered answer all agree."""
    if response.get("answerStatus") != "answered":
        return []
    reference = response.get("reference", {})
    chunks = reference.get("chunks", [])
    spans = reference.get("spans", [])
    if not spans or reference.get("total") != len(chunks):
        return []
    rendered = []
    cited = set()
    for span in spans:
        index, start, end = (span.get(key) for key in ("citationIndex", "start", "end"))
        if any(type(value) is not int for value in (index, start, end)):
            return []
        if not 0 <= index < len(chunks) or not 0 <= start < end:
            return []
        # Java strings and browser JavaScript use UTF-16 indices.
        source = source_text(chunks[index]).encode("utf-16-le")
        if end * 2 > len(source):
            return []
        try:
            text = source[start * 2:end * 2].decode("utf-16-le")
        except UnicodeDecodeError:
            return []
        if not text.strip() or text != span.get("text"):
            return []
        cited.add(index)
        safe_text = text.replace("[ID:", "［ID:")
        rendered.append(f"{safe_text} [ID:{index}]")
    if cited != set(range(len(chunks))) or "\n".join(rendered) != response.get("answer"):
        return []
    return spans


def explicit_refusal(response):
    reference = response.get("reference", {})
    answer = response.get("answer", "").strip()
    return (response.get("answerStatus") == "insufficient_evidence" and answer in REFUSALS
            and reference.get("total") == 0 and reference.get("chunks") == [] and reference.get("spans") == [])


def extractive_context(boundary):
    messages = boundary.get("messages", [])
    if not messages:
        return ""
    payload = json.loads(messages[-1]["content"])
    if "candidateExcerpts" in payload:
        return "\n".join(passage[2] for passage in payload["candidateExcerpts"])
    return "\n".join(passage["text"] for passage in payload["passages"])
