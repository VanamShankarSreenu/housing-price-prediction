from scripts import  CombinedAttributesAdder
import unittest
from ta_lib.core.api import (load_dataset,create_context)
import os.path as op
from sklearn.impute import SimpleImputer
import pandas as pd

class Test_pipeline(unittest.TestCase):
    def test_customadder(self): 
        config_path = op.join('conf', 'config.yml')
        context = create_context(config_path)
        train_X = load_dataset(context, 'train/house/features')
        train_y = load_dataset(context, 'train/house/target')
        num_attrib = train_X.select_dtypes('number').columns
        train_X = train_X[num_attrib]
        impute = SimpleImputer(strategy="median")
        train_X = impute.fit_transform(train_X)
        feat_adder = CombinedAttributesAdder()
        output = feat_adder.fit_transform(train_X,train_y)
        train_X = pd.DataFrame(data=train_X,columns=num_attrib)
        extra_attrib = ["rooms_per_household","population_per_household",
                "bedrooms_per_room"]
        att = list(num_attrib)+extra_attrib
        output = pd.DataFrame(data=output,columns=att)
        assert "rooms_per_household" in output
        assert "population_per_household" in output
        assert "bedrooms_per_room" in output
        self.assertEqual(sum(train_X["population"]/train_X["households"]), sum(output["population_per_household"]))
        self.assertEqual(sum(train_X["total_rooms"]/train_X["households"]), sum(output["rooms_per_household"]))
        self.assertEqual(sum(train_X["total_bedrooms"]/train_X["total_rooms"]), sum(output["bedrooms_per_room"]))


if __name__ == '__main__':
    unittest.main()