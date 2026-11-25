import copy
import unittest

from qa_checks import explicit_refusal, verified_spans


class QaChecksTest(unittest.TestCase):
    def test_utf16_offsets_and_rendered_text_must_both_match(self):
        response = {
            "answerStatus": "answered",
            "answer": "三天。 [ID:0]",
            "reference": {"total": 1, "chunks": [{"content": "🙂三天。"}],
                          "spans": [{"citationIndex": 0, "start": 2, "end": 5, "text": "三天。"}]},
        }
        self.assertTrue(verified_spans(response))
        answer_changed = copy.deepcopy(response)
        answer_changed["answer"] = "两天。 [ID:0]"
        self.assertFalse(verified_spans(answer_changed))
        bad_offset = copy.deepcopy(response)
        bad_offset["reference"]["spans"][0]["start"] = 1
        self.assertFalse(verified_spans(bad_offset))

    def test_refusal_cannot_contain_a_guess_or_citation(self):
        response = {"answerStatus": "insufficient_evidence", "answer": "资料不足，无法回答。",
                    "reference": {"total": 0, "chunks": [], "spans": []}}
        self.assertTrue(explicit_refusal(response))
        self.assertFalse(explicit_refusal(dict(response, answer="猜测是三天。资料不足，无法回答。")))
        self.assertFalse(explicit_refusal(dict(response, reference={"total": 1, "chunks": [{}], "spans": []})))


if __name__ == "__main__":
    unittest.main()
