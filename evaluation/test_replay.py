import copy,unittest
from replay import candidate_messages
class ReplayTest(unittest.TestCase):
    def test_changes_instruction_but_freezes_knowledge_citation_tail_and_conversation(self):
        source=[{'role':'system','content':'old instruction\n      Here is the knowledge base:\nID: 0\nEvidence.\nCitation rules.'},{'role':'user','content':'Question'}]
        before=copy.deepcopy(source);candidate,sha=candidate_messages(source)
        marker='      Here is the knowledge base:\n'
        self.assertEqual(source,before)
        self.assertEqual(source[0]['content'].split(marker)[1],candidate[0]['content'].split(marker)[1])
        self.assertEqual(source[1:],candidate[1:]);self.assertEqual(64,len(sha))
    def test_unknown_format_fails_closed(self):
        with self.assertRaises(ValueError):candidate_messages([{'role':'system','content':'no captured context'}])
if __name__=='__main__':unittest.main()
