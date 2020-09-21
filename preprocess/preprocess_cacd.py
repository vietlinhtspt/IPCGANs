# from mtcnn.mtcnn import MTCNN
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm
from skimage import transform as trans
import time

#define read path
root_data_dir="/home/linhnv/Downloads"
image_root_dir=os.path.join(root_data_dir,"CACD2000")
#define store path
store_root_dir="/home/linhnv/Downloads/ExperimentDataset/CACD"
store_image_dir=os.path.join(store_root_dir,"CACD2000")
if os.path.exists(store_image_dir) is False:
    os.makedirs(store_image_dir)

#define some param for mtcnn
src = np.array([
 [30.2946, 51.6963],
 [65.5318, 51.5014],
 [48.0252, 71.7366],
 [33.5493, 92.3655],
 [62.7299, 92.2041] ], dtype=np.float32 )
threshold = [0.6,0.7,0.9]
factor = 0.85
minSize=20
imgSize=[120, 100]
# detector=MTCNN(steps_threshold=threshold,scale_factor=factor,min_face_size=minSize)
detector=MTCNN()
#align,crop and resize
keypoint_list=['left_eye','right_eye','nose','mouth_left','mouth_right']

for idx, filename in enumerate(tqdm(os.listdir(image_root_dir))):
    # start = time.time()
    dst = []
    filepath=os.path.join(image_root_dir,filename)
    storepath=os.path.join(store_image_dir,filename)
    
    image = Image.open(filepath)
    #Image.fromarray(npimage.astype(np.uint8)).show()
    
    img_face = detector.align_and_take_one(image)#if more than one face is detected, [0] means choose the first face
    # print(time.time() - start)
    if img_face is None:
        continue
    img_face.resize((400, 400))
    img_face.save(storepath)
    npimage=np.array(image)
    # if len(dictface_list)>1:
    #     boxs=[]
    #     for dictface in dictface_list:
    #         boxs.append(dictface['box'])
    #     center=np.array(npimage.shape[:2])/2
    #     boxs=np.array(boxs)
    #     face_center_y=boxs[:,0]+boxs[:,2]/2
    #     face_center_x=boxs[:,1]+boxs[:,3]/2
    #     face_center=np.column_stack((np.array(face_center_x),np.array(face_center_y)))
    #     distance=np.sqrt(np.sum(np.square(face_center - center),axis=1))
    #     min_id=np.argmin(distance)
    #     dictface=dictface_list[min_id]
    # else:
    #     if len(dictface_list)==0:
    #         continue
    #     else:
    #         dictface=dictface_list[0]
    # face_keypoint = dictface['keypoints']
    # for keypoint in keypoint_list:
    #     dst.append(face_keypoint[keypoint])
    # dst = np.array(dst).astype(np.float32)
    # tform = trans.SimilarityTransform()
    # tform.estimate(dst, src)
    # M = tform.params[0:2, :]
    # warped = cv2.warpAffine(npimage, M, (imgSize[1], imgSize[0]), borderValue=0.0)
    # warped=cv2.resize(warped,(400,400))
    # Image.fromarray(warped.astype(np.uint8)).save(storepath)
    # print(time.time() - start)
    