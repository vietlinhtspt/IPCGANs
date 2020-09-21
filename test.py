import glob
from os import listxattr
import os
from preprocess.gentraingroup import test_txt

list_path_imgs = glob.glob(f"/home/linhnv/Downloads/ExperimentDataset/CACD/CACD2000/*")
test_txt = []
for path_img in list_path_imgs:
    if os.path.basename(path_img).startswith("55_"):
        test_txt.append(os.path.basename(path_img) + "\n")
    

with open('test.txt','w') as f:
    f.writelines(test_txt)