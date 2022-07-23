from unittest import TestCase
from datetime import datetime
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import batch

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

class Test(TestCase):
    def test_prepare_data(self):
        data = [
            (None, None, dt(1, 2), dt(1, 10)),
            (1, 1, dt(1, 2), dt(1, 10)),
            (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
            (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
        ]
        columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
        df = pd.DataFrame(data, columns=columns)
        prepared_df: pd.DataFrame = batch.prepare_data(df, ['PUlocationID', 'DOlocationID'])

        self.assertIn("duration", prepared_df.columns)

        self.assertGreaterEqual(prepared_df["duration"].min(), 1)
        self.assertLessEqual(prepared_df["duration"].max(), 60)
        self.assertEqual(prepared_df.shape[0], 2)

        expected = {
            "PUlocationID": {0: '-1', 1: '1'},
            'DOlocationID': {0: '-1', 1: '1'},
            'pickup_datetime' : {0: dt(1, 2), 1:  dt(1, 2)},
            'dropOff_datetime' : {0: dt(1, 10), 1:  dt(1, 10)},
            "duration" : {0: 8.0, 1: 8.0}
        }
        expected = pd.DataFrame(expected)
        self.assertDictEqual(expected[columns].to_dict(), prepared_df[columns].to_dict())
        for exp_item, act_item in zip(expected['duration'],prepared_df['duration'] ):
            self.assertAlmostEqual(exp_item, act_item)
