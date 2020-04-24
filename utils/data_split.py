import glob
import os
import numpy as np
import random

def split_data(dirA, dirB, percent_dict = {'train_perc': 0.8, 'val_perc':0.1, 'test_perc':0.1}):
    train_perc = percent_dict['train_perc']
    val_perc = percent_dict['val_perc']
    test_perc = percent_dict['test_perc']

    camA_files = sorted(
        glob.glob(
            os.path.join(dirA, '*.png')
        )
    )
    camB_files = sorted(
        glob.glob(
            os.path.join(dirB, '*.png')
        )
    )

    person_ids  = [int(x.split('/')[-1][8:13]) - 1 for x in (camA_files)] + [int(x.split('/')[-1][8:13]) -1 for x in (camB_files)] 
    all_ids = np.unique(person_ids)

    train_ids = all_ids[0:int(len(all_ids)*train_perc)]
    val_ids = all_ids[int(len(all_ids)*train_perc):int(len(all_ids)*(train_perc+val_perc))]
    test_ids = all_ids[int(len(all_ids)*(train_perc+val_perc)):int(len(all_ids)*(train_perc+val_perc+test_perc))]
    
    # training data
    dataA = []
    dataB = []
    identA = []
    identB = []

    for file in camA_files:
      dataA.append(file)
      identA.append(int(file.split('/')[-1][8:13]) - 1)

    for file in camB_files:
      dataB.append(file)
      identB.append(int(file.split('/')[-1][8:13]) -1)

    # validation data
    val_dataA = np.array(dataA)[np.isin(identA,val_ids)]
    val_identA = np.array(identA)[np.isin(identA,val_ids)]
    val_dataB = np.array(dataB)[np.isin(identB,val_ids)]
    val_identB = np.array(identB)[np.isin(identB,val_ids)]

    # test data
    test_dataA = np.array(dataA)[np.isin(identA,test_ids)]
    test_identA = np.array(identA)[np.isin(identA,test_ids)]
    test_dataB = np.array(dataB)[np.isin(identB,test_ids)]
    test_identB = np.array(identB)[np.isin(identB,test_ids)]

    dataA = np.array(dataA)[np.isin(identA,train_ids)]
    identA = np.array(identA)[np.isin(identA,train_ids)]
    dataB = np.array(dataB)[np.isin(identB,train_ids)]
    identB = np.array(identB)[np.isin(identB,train_ids)]

    return all_ids, dataA, identA, dataB, identB, val_dataA, val_identA, val_dataB, val_identB, test_dataA, test_identA, test_dataB, test_identB 



