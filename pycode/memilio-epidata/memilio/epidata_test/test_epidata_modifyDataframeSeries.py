#############################################################################
# Copyright (C) 2020-2021 German Aerospace Center (DLR-SC)
#
# Authors: Sascha Korf, Patrick Lenz
#
# Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#############################################################################

import unittest
import pandas as pd
from pyfakefs import fake_filesystem_unittest
from memilio.epidata import modifyDataframeSeries as mDfS

class Test_modifyDataframeSeries(fake_filesystem_unittest.TestCase):
    test_df1 = pd.DataFrame(
        {
         'Date':
         ['2021-01-06', '2021-01-06', '2021-01-06', '2021-01-07', '2021-01-07',
          '2021-01-07', '2021-01-08', '2021-01-08', '2021-01-08', '2021-01-09',
          '2021-01-09', '2021-01-09', '2021-01-10', '2021-01-10',
          '2021-01-10'],
         'test_col1': [12, 3, 6, 0, 3, 1, 4, 7, 11, 15, 19, 19, 27, 13, 5],
         'test_col2': ['a', 'x', 't', 'a', 'b', 'a', 'x', 't', 'a', 'b', 'a', 'x', 't', 'a', 'b'],
         'test_col3': [1, 0, 1, 9, 4, 3, 2, 1, 1, 1, 0, 6, 5, 3, 1],
         'ID': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]})
    test_df2 = pd.DataFrame(
        {
         'Date':
         ['2021-01-06', '2021-01-06', '2021-01-06', '2021-01-07', '2021-01-07',
          '2021-01-07', '2021-01-08', '2021-01-08', '2021-01-08', '2021-01-09',
          '2021-01-09', '2021-01-09', '2021-01-13', '2021-01-13',
          '2021-01-13'],
         'test_col1': [12, 3, 6, 0, 3, 1, 4, 7, 11, 15, 19, 19, 27, 13, 5],
         'test_col2': ['a', 'x', 't', 'a', 'b', 'a', 'x', 't', 'a', 'b', 'a', 'x', 't', 'a', 'b'],
         'test_col3': [1, 0, 1, 9, 4, 3, 2, 1, 1, 1, 0, 6, 5, 3, 1],
         'ID': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]})


    def setUp(self):
        self.setUpPyfakefs()

    def test_impute_and_reduce_df(self):

        group_by_cols = {'ID': sorted(set(self.test_df1['ID'].unique()))}
        mod_cols = ['test_col1', 'test_col3']

        # test impute forward and fill dates with moving average = 3
        df = mDfS.impute_and_reduce_df(
            self.test_df1, group_by_cols, mod_cols, impute='forward',
            moving_average=3, min_date='2021-01-05', max_date='2021-01-11',
            start_w_firstval=False)
        # test that values at new start days are zero since start_w_firstval=False
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-05") & (df['ID'] == 2.0)]['test_col1'].item(), 0)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-05") & (df['ID'] == 2.0)]['test_col3'].item(), 0)
        # test that the values at first original date are obtained by the value itself plus the value right of it divided by 3
        # (6 + 1) / 3 = 2 + 2 / 3
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-06") & (df['ID'] == 3.0)]['test_col1'].item(), 2 + 1 / 3)
        # (3 + 1) / 3 = 1 + 1 / 3
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-06") & (df['ID'] == 3.0)]['test_col3'].item(), 1 + 1 / 3)
        
        # check that last entry of original frame is computed by the value left of it and twice its value since impute='forward'
        # (15 + 27 + 27) / 3 = 23
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-10") & (df['ID'] == 1.0)]['test_col1'].item(), 23)
        # (1 + 5 + 5) / 3 = 3 + 2 / 3
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-10") & (df['ID'] == 1.0)]['test_col3'].item(), 3 + 2/ 3)
        
        # check that new date gets imputed the value the column had the day before because impute='forward'
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-11") & (df['ID'] == 3.0)]['test_col3'].item(), 1)

        # test impute zeros with moving average = 3
        df = mDfS.impute_and_reduce_df(
            self.test_df1, group_by_cols, mod_cols, impute='zeros',
            moving_average=3, min_date='2021-01-05', max_date='2021-01-11',
            start_w_firstval=False)
        # test that values at new start days are zero since start_w_firstval=False
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-05") & (df['ID'] == 2.0)]['test_col3'].item(), 0)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-05") & (df['ID'] == 2.0)]['test_col1'].item(), 0)

        # test that the values at first original date are obtained by the value itself plus the value right of it divided by 3
        # (6 + 1) / 3 = 2 + 2 / 3
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-06") & (df['ID'] == 3.0)]['test_col1'].item(), 2 + 1 / 3)
        # (3 + 1) / 3 = 1 + 1 / 3
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-06") & (df['ID'] == 3.0)]['test_col3'].item(), 1 + 1 / 3)
        
        # check that last entry of original frame is computed by the value left of it and the value itself because impute = "zeros"
        # (15 + 27) / 3 = 14
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-10") & (df['ID'] == 1.0)]['test_col1'].item(), 14)
        # (1 + 5) / 3 = 2
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-10") & (df['ID'] == 1.0)]['test_col3'].item(), 2)

        # check that new date gets imputed 0 because impute = "zeros"
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-11") & (df['ID'] == 3.0)]['test_col3'].item(), 0)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-11") & (df['ID'] == 3.0)]['test_col1'].item(), 0)

        # test fill missing dates moving average = 4
        # if moving average is an even number it always should calculate with one more earlier date
        df = mDfS.impute_and_reduce_df(
            self.test_df2, group_by_cols, mod_cols, impute='forward',
            moving_average=4, min_date='2021-01-06', max_date='2021-01-13',
            start_w_firstval=False)
        # test that the values at first original date arent changed since there is no value left of it
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-06") & (df['ID'] == 2.0)]['test_col3'].item(), 0)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-06") & (df['ID'] == 2.0)]['test_col1'].item(), 3)
        # test that a value is computed by two values left of it, the value itself and the value right of it divided by 4
        # (6 + 1 + 11 + 19) / 4 = 9 + 1 / 4
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-08") & (df['ID'] == 3.0)]['test_col1'].item(), 9 + 1 / 4)
        # (1 + 3 + 1 + 6) = 2 + 3 / 4
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-08") & (df['ID'] == 3.0)]['test_col3'].item(), 2 + 3 / 4)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-13") & (df['ID'] == 3.0)]['test_col3'].item(), 1)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-13") & (df['ID'] == 3.0)]['test_col1'].item(), 5)
        # test that the first of three missing dates is computed by three times the value left of it plus the penultimate value devided by 4
        # (19 + 19 + 19 + 11) / 4 = 17
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-10") & (df['ID'] == 3.0)]['test_col1'].item(), 17)
        # (6 + 6 + 6 + 1) / 4 = 4 + 3 / 4
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-10") & (df['ID'] == 3.0)]['test_col3'].item(), 4 + 3/4)

        # test mod_cols = ['test_col1']. test_col3 should not be modified
        mod_cols = ['test_col1']
        df = mDfS.impute_and_reduce_df(
            self.test_df2, group_by_cols, mod_cols, impute='forward',
            moving_average=4, min_date='2021-01-06', max_date='2021-01-13',
            start_w_firstval=False)
        # test same tests as in the previous test with moving average = 4
        # 'test_col1' should be same same as in the previous test
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-06") & (df['ID'] == 2.0)]['test_col1'].item(), 3)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-08") & (df['ID'] == 3.0)]['test_col1'].item(), 9 + 1 / 4)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-13") & (df['ID'] == 3.0)]['test_col1'].item(), 5)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-10") & (df['ID'] == 3.0)]['test_col1'].item(), 17)
        # 'test_col3' should not be changed
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-06") & (df['ID'] == 2.0)]['test_col3'].item(), 0)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-08") & (df['ID'] == 3.0)]['test_col3'].item(), 1)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-13") & (df['ID'] == 3.0)]['test_col3'].item(), 1)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-10") & (df['ID'] == 3.0)]['test_col3'].item(), 6)

        mod_cols = ['test_col1', 'test_col3']
        # test start date higher than end date
        # empty dataframe should be returned
        df = mDfS.impute_and_reduce_df(
            self.test_df1, group_by_cols, mod_cols, impute='forward',
            moving_average=4, min_date='2021-01-13', max_date='2021-01-06',
            start_w_firstval=False)
        edf = pd.DataFrame()
        self.assertEqual(len(edf),len(df))

        # test start_w_firstval = True
        df = mDfS.impute_and_reduce_df(
            self.test_df1, group_by_cols, mod_cols, impute='forward',
            moving_average=3, min_date='2021-01-05', max_date='2021-01-11',
            start_w_firstval=True)
        # test that values at new start days are the same as in the first original date since start_w_firstval=True
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-05") & (df['ID'] == 2.0)]['test_col1'].item(), 3)
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-05") & (df['ID'] == 2.0)]['test_col3'].item(), 0)
        # test that the values at first original date are twice the value itself plus the value right of it divided by 3 since start_w_firstval=True
        # (6 + 6 + 1) / 3 = 4 + 2 / 3
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-06") & (df['ID'] == 3.0)]['test_col1'].item(), 4 + 1 / 3)
        # (1+ 1 + 3) / 3 = 1 + 2 / 3
        self.assertAlmostEqual(df[(df['Date'] == "2021-01-06") & (df['ID'] == 3.0)]['test_col3'].item(), 1 + 2 / 3)


if __name__ == '__main__':
    unittest.main()
