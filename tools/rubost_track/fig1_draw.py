import cv2
from glob import glob
import numpy as np
import os

# set_name='Basketball'  # 485,
# set_name='Girl2'
# frame_num = 150
# set_name='Board'  # 110,
# set_name='MotorRolling' 150
set_name='Walking2'
frame_num = 1


frame_num_end=1000

res_path='/home/rislab/Workspace/pysot/rb_result/'+ set_name
image_path='/home/rislab/Workspace/pysot/testing_dataset/OTB100/'+set_name+'/img'
gt_rects=np.loadtxt(image_path.replace('img', 'groundtruth_rect.txt'),delimiter=',',dtype='int')


results = glob(os.path.join(res_path, '*.txt'))
results = sorted(results, key=lambda x: int(x.split('/')[-1].split('.')[0]))

images = glob(os.path.join(image_path, '*.jp*'))
images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))

for i in range(frame_num,frame_num_end):
    frame = cv2.imread(images[i])
    bbox_gt=gt_rects[i-1,:]
    bbox_1=np.loadtxt(results[0])[i,:].astype(np.int)
    bbox_2=np.loadtxt(results[1])[i,:].astype(np.int)
    bbox_3 = np.loadtxt(results[2])[i, :].astype(np.int)
    bbox_4 = np.loadtxt(results[3])[i, :].astype(np.int)

    cv2.rectangle(frame, (bbox_gt[0], bbox_gt[1]),
                  (bbox_gt[0] + bbox_gt[2], bbox_gt[1] + bbox_gt[3]),
                  (0, 255, 0), 2)
    cv2.rectangle(frame, (bbox_1[0], bbox_1[1]),
                  (bbox_1[0] + bbox_1[2], bbox_1[1] + bbox_1[3]),
                  (255, 0, 0), 2)
    cv2.rectangle(frame, (bbox_2[0], bbox_2[1]),
                  (bbox_2[0] + bbox_2[2], bbox_2[1] + bbox_2[3]),
                  (255, 0, 255), 2)
    cv2.rectangle(frame, (bbox_3[0], bbox_3[1]),
                  (bbox_3[0] + bbox_3[2], bbox_3[1] + bbox_3[3]),
                  (0, 255, 255), 2)
    cv2.rectangle(frame, (bbox_4[0], bbox_4[1]),
                  (bbox_4[0] + bbox_4[2], bbox_4[1] + bbox_4[3]),
                  (0, 0, 255), 2)

    strshow = 'frame: ' + str(int(i))
    frame = cv2.putText(frame, strshow, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1., (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('', frame)
    cv2.waitKey(0)
    print('')
    cv2.imwrite(res_path+'/'+set_name+'.jpg',frame)
