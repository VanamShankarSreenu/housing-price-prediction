from scripts import  CombinedAttributesAdder
import unittest
from sklearn.utils.estimator_checks import check_estimator
from ta_lib.core.api import load_dataset
class Test_pipeline(unittest.TestCase):
    def test_customadder(self):
        self.assertEqual(check_estimator(CombinedAttributesAdder()),True)
        
        