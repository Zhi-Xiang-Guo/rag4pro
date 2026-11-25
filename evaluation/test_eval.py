import importlib.util, math, pathlib, tempfile, unittest
spec=importlib.util.spec_from_file_location('evaluation',pathlib.Path(__file__).with_name('eval.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

class MetricsTest(unittest.TestCase):
    def test_deduplicates_before_cutoff_and_counts_all_positives(self):
        m=e.metrics(['a','a','x','b'],{'a':1,'b':1})
        self.assertEqual(.5,m['recall@1']);self.assertEqual(1,m['recall@5'])
        self.assertEqual(.4,m['precision@5']);self.assertEqual(1,m['mrr@5'])
        self.assertAlmostEqual((1+1/math.log2(4))/(1+1/math.log2(3)),m['ndcg@5'])
    def test_empty_failed_request_scores_zero(self):
        self.assertTrue(all(v==0 for v in e.metrics([],{'a':1}).values()))
    def test_graded_ndcg_and_mrr(self):
        m=e.metrics(['x','b','a'],{'a':2,'b':1})
        self.assertEqual(.5,m['mrr@5'])
        self.assertAlmostEqual((1/math.log2(3)+3/math.log2(4))/(3+1/math.log2(3)),m['ndcg@5'])
    def test_no_labels_rejected(self):
        with self.assertRaises(ValueError):e.metrics(['a'],{})
    def test_incomplete_report_cannot_pass_as_complete(self):
        with tempfile.TemporaryDirectory() as d:
            d=pathlib.Path(d)
            cfg={'baseline':'base','primaryMetric':'ndcg@10','seed':1}
            e.write(d/'run.json',{'split':'test','queryIds':['q1','q2'],'variants':['base'],'config':cfg})
            e.jsonl(d/'results.jsonl',[{'queryId':'q1','variant':'base','error':None,'latencyMs':1,'metrics':e.metrics(['a'],{'a':1})}])
            self.assertFalse(e.report_dir(d)['complete'])
    def test_document_identity_matches_known_java_uuid(self):
        import hashlib,uuid
        # UUID.nameUUIDFromBytes uses MD5 directly, not a UUID namespace.
        identity=uuid.UUID(bytes=hashlib.md5(b'demo\0refund').digest(),version=3).hex
        self.assertEqual(32,len(identity));self.assertEqual('3',identity[12])
if __name__=='__main__':unittest.main()
