import importlib.util,pathlib,unittest,tempfile,csv
s=importlib.util.spec_from_file_location('release_gate',pathlib.Path(__file__).with_name('release_gate.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class ReleaseTest(unittest.TestCase):
    def test_good_retrieval_does_not_mask_hallucination_or_pending_review(self):
        qa={'complete':True,'variants':{'candidate':{'errors':0,'factPatternPassRate':1,'refusalPatternPassRate':0}}}
        result=m.evaluate({'passed':True},qa,{'runs':[{'otherFailures':0}]})
        self.assertFalse(result['passed']);self.assertEqual(2,len(result['failures']))
    def test_upstream_error_blocks_even_if_rule_checks_pass(self):
        qa={'complete':True,'variants':{'candidate':{'errors':1,'factPatternPassRate':1,'refusalPatternPassRate':1}}}
        self.assertFalse(m.evaluate({'passed':True},qa,{'runs':[{'otherFailures':0}]},reviewed=True)['passed'])
    def test_selected_candidate_can_pass_without_requiring_old_baseline_to_pass(self):
        good={'errors':0,'factPatternPassRate':1,'refusalPatternPassRate':1}
        bad={'errors':0,'factPatternPassRate':.4,'refusalPatternPassRate':0}
        qa={'complete':True,'variants':{'original':bad,'candidate':good}}
        self.assertTrue(m.evaluate({'passed':True},qa,{'runs':[{'otherFailures':0}]},reviewed=True,selected='candidate')['passed'])
    def test_review_is_bound_to_actual_answer_and_complete_scores(self):
        record={'id':'q','variant':'candidate','question':'question','response':{'answer':'captured answer'},'unanswerable':False}
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'review.csv'
            fields=['id','variant','question','answer','correctness_0_1','faithfulness_0_1','citationSupport_0_1','reviewer']
            row=dict(zip(fields,['q','candidate','question','captured answer','1','1','1','fixture-reviewer']))
            def write():
                with open(path,'w') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerow(row)
            write();self.assertTrue(m.validate_review(path,[record],'candidate')['passed'])
            row['answer']='different answer';write()
            self.assertFalse(m.validate_review(path,[record],'candidate')['passed'])
    def test_missing_selected_variant_is_rejected(self):
        self.assertFalse(m.evaluate({'passed':True},{'complete':True,'variants':{}},{'runs':[{'otherFailures':0}]},reviewed=True,selected='missing')['passed'])
if __name__=='__main__':unittest.main()
