import unittest
import numpy as np
import pandas as pd
from spend_course10.item_season_features import ITEM_SEASON_COLUMNS,item_season_features


def lines(rows):
    a=pd.DataFrame(rows,columns=['customer_id','item_id','event_ts']);a.event_ts=pd.to_datetime(a.event_ts);return a


class ItemSeasonTests(unittest.TestCase):
    def test_global_old_product_season_transfers_to_new_customer(self):
        a=lines([('old','A',f'2017-09-{d:02d}') for d in [2,3,4,5]]+[('new','A','2018-08-20')])
        obj=pd.DataFrame({'customer_id':['new'],'as_of_date':pd.to_datetime(['2018-09-01'])})
        x=item_season_features(obj,a,'2016-12-01',max_items=4)
        self.assertAlmostEqual(float(x[0,ITEM_SEASON_COLUMNS.index('item_year_season')]),float(np.log(9/5)),places=6)
        self.assertEqual(x[0,ITEM_SEASON_COLUMNS.index('item_year_seen_fraction')],1)

    def test_future_poison_and_new_future_codes_do_not_change_features(self):
        a=lines([('old','B','2017-09-02'),('new','B','2018-08-20')])
        obj=pd.DataFrame({'customer_id':['new'],'as_of_date':pd.to_datetime(['2018-09-01'])})
        poison=pd.concat([a,lines([('new','A','2018-09-01'),('other','B','2018-09-01'),('other','A','2020-01-01')])],ignore_index=True)
        np.testing.assert_array_equal(item_season_features(obj,a,'2016-12-01',max_items=4),item_season_features(obj,poison,'2016-12-01',max_items=4))

    def test_missing_year_coverage_and_known_zero_remain_distinct(self):
        a=lines([('new','A','2018-08-20')]);obj=pd.DataFrame({'customer_id':['new'],'as_of_date':pd.to_datetime(['2018-09-01'])})
        x=item_season_features(obj,a,'2018-01-01',max_items=4)
        y=item_season_features(obj,a,'2016-12-01',max_items=4)
        col=ITEM_SEASON_COLUMNS.index('item_year30')
        self.assertTrue(np.isnan(x[0,col]));self.assertEqual(y[0,col],0)


if __name__=='__main__':unittest.main()
