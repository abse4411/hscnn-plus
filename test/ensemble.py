import os
import numpy as np
import h5py
from utils import save_matv73 



mat_path1 = './test_results1'
mat_path2 = './test_results2'
mat_path3 = './test_results3'
save_path = './final_results'

for mat_name in sorted(os.listdir(mat_path1)):
    print (mat_name)
    mat_path_name1 = os.path.join(mat_path1, mat_name)
    hf1 = h5py.File(mat_path_name1)
    data1 = hf1.get('rad')
    res1 = np.transpose(np.array(data1),[2,1,0])

    mat_path_name2 = os.path.join(mat_path2, mat_name)
    hf2 = h5py.File(mat_path_name2)
    data2 = hf2.get('rad')
    res2 = np.transpose(np.array(data2),[2,1,0])

    mat_path_name3 = os.path.join(mat_path3, mat_name)
    hf3 = h5py.File(mat_path_name3)
    data3 = hf3.get('rad')
    res3 = np.transpose(np.array(data3),[2,1,0])

    res = 0.4*res1+ 0.3*res2+ 0.3*res3
    mat_dir= os.path.join(save_path, mat_name)
    save_matv73(mat_dir, 'rad', res)
