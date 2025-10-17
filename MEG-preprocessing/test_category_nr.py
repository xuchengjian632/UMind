"""
Used to collecte the test category number for discard
"""


import numpy as np
import pandas as pd

nSub = 1
pd_label = pd.read_csv('/media/siat/disk1/BCI_data/THINGS-MEG/label_csv/sample_attributes_P{:01d}.csv'.format(nSub))

trial_type = pd_label['trial_type'].values
things_category = pd_label['things_category_nr'].values # float 64
test_image = pd_label['test_image_nr'].values

test_idx = np.sort(np.where(trial_type == 'test')[0])

test_category = things_category[test_idx]
test_id = np.unique(test_category)

print('The length of test_id is: ', len(test_id))
print('The test id is: ', test_id)

np.save('./data/THINGS-MEG_test_category.npy', test_id)

