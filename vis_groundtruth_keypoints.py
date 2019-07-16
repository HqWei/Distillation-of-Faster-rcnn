import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import cv2

from PIL import Image
from matplotlib import pyplot as plt
import argparse
import numpy as np
global false, null, true
false =False
true=True



# def align_boxes(img_path,instances,target_dir,box,iou,w,h):
'''
:param img_path:
:param instances:
:return:
'''
from matplotlib import pyplot, patches
import numpy as np
import math


# PERSON_KEYPOINTS = [
#     "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
#     "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
#     "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
#     "right_ankle"
# ]

# KEYPOINT_PAIRS = [(i, i + 1) for i in range(1, 16, 2)]

# for visualization
# KEYP_LINES = [
#     [PERSON_KEYPOINTS.index('left_eye'), PERSON_KEYPOINTS.index('right_eye')],
#     [PERSON_KEYPOINTS.index('left_eye'), PERSON_KEYPOINTS.index('nose')],
#     [PERSON_KEYPOINTS.index('right_eye'), PERSON_KEYPOINTS.index('nose')],
#     [PERSON_KEYPOINTS.index('right_eye'), PERSON_KEYPOINTS.index('right_ear')],
#     [PERSON_KEYPOINTS.index('left_eye'), PERSON_KEYPOINTS.index('left_ear')],
#     [PERSON_KEYPOINTS.index('right_shoulder'), PERSON_KEYPOINTS.index('right_elbow')],
#     [PERSON_KEYPOINTS.index('right_elbow'), PERSON_KEYPOINTS.index('right_wrist')],
#     [PERSON_KEYPOINTS.index('left_shoulder'), PERSON_KEYPOINTS.index('left_elbow')],
#     [PERSON_KEYPOINTS.index('left_elbow'), PERSON_KEYPOINTS.index('left_wrist')],
#     [PERSON_KEYPOINTS.index('right_hip'), PERSON_KEYPOINTS.index('right_knee')],
#     [PERSON_KEYPOINTS.index('right_knee'), PERSON_KEYPOINTS.index('right_ankle')],
#     [PERSON_KEYPOINTS.index('left_hip'), PERSON_KEYPOINTS.index('left_knee')],
#     [PERSON_KEYPOINTS.index('left_knee'), PERSON_KEYPOINTS.index('left_ankle')],
#     [PERSON_KEYPOINTS.index('right_shoulder'), PERSON_KEYPOINTS.index('left_shoulder')],
#     [PERSON_KEYPOINTS.index('right_hip'), PERSON_KEYPOINTS.index('left_hip')],
#     [PERSON_KEYPOINTS.index('left_ear'), PERSON_KEYPOINTS.index('left_shoulder')],
#     [PERSON_KEYPOINTS.index('right_ear'), PERSON_KEYPOINTS.index('right_shoulder')],
#     [PERSON_KEYPOINTS.index('left_shoulder'), PERSON_KEYPOINTS.index('left_hip')],
#     [PERSON_KEYPOINTS.index('right_shoulder'), PERSON_KEYPOINTS.index('right_hip')],
# ]


'''
14:

'''
KEYPOINT_COLORS = ['magenta', 'cyan', 'yellow', 'green',
                   'lime', 'blue', 'purple', 'orange',
                   'white', 'lightcoral', 'lime', 'olive',
                   'steelblue', 'red', 'gold', 'navy',
                   'dodgerblue', 'mediumaquamarine', 'black', 'gray']
# KEYPOINT_PAIRS = \
#     [
#         (0,1),(0,2),(0,5),
#         (1,2),(1,5),(1,8),(1,11),
#         (2,3),(3,4),(5,6),(6,7),
#         (8,9),(9,10),(11,12),(12,13)
#     ]

'''
11:
PERSON_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]

'''
KEYPOINT_PAIRS = \
    [
        (0,1),(0,2),
        (1,3),(3,5),(5,7),(7,9),
        (2,4),(4,6),(6,8),(8,10)
    ]


def vis_one_img(img_root,grountruth_file,vis_num):
    f=open(grountruth_file,'r')
    lines=f.readlines()
    line=lines[vis_num-1]
    line_dic=eval(line)

    img_name=line_dic['filename']
    print(img_name,flush=True)
    img_position = os.path.join(img_root,img_name)
   
    image = cv2.imread(img_position, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.ion()
    fig = plt.figure(figsize=(5, 5))
    #
    plt.imshow(img)
    currentAxis = plt.gca()
    instances=line_dic['instances']
    for i in range(len(instances)):
        roi = instances[i]['bbox']
        if instances[i]['label'] == 1:
            if instances[i]['is_ignored']:
                currentAxis.add_patch(plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                                                    roi[3] - roi[1], fill=False,
                                                    edgecolor='y', linewidth=1.5))
                # if i==38:
                currentAxis.text(float(roi[0]), roi[1], i,
                                 color='b',fontsize=10,bbox={'facecolor': 'g','alpha': 0.001})
            else:
                currentAxis.add_patch(plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                                                    roi[3] - roi[1], fill=False,
                                                    edgecolor='r', linewidth=2.5))
                # if i == 38:
                currentAxis.text(float(roi[0]), roi[1], i,
                             color='r',fontsize=10,bbox={'facecolor': 'g','alpha': 0.001})
        else:
            if instances[i]['is_ignored']:
                currentAxis.add_patch(plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                                                    roi[3] - roi[1], fill=False,
                                                    edgecolor='g', linewidth=1.5))
            else:
                currentAxis.add_patch(plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                                                    roi[3] - roi[1], fill=False,
                                                    edgecolor='w', linewidth=1.5))
        keypoints=instances[i]['keypoints']
        if keypoints is not None and len(keypoints) > 0:
            # im = vis_keypoints(im, keypoints[i])


            colormap_index = np.linspace(0, 1, len(KEYPOINT_PAIRS))
            # for i in range(18):
            keypoints_2d = keypoints
            for corner_xy, color in zip(keypoints_2d, KEYPOINT_COLORS):
                print(corner_xy)
                # corner_xy=int(corner_xy)
                if corner_xy[2]>0:
                    currentAxis.add_patch(patches.Circle(corner_xy, radius=4, fill=True, edgecolor=color))

            pts = np.array(keypoints)

            joint_visible = pts[:, 2] > 0

            for cm_ind, jp in zip(colormap_index, KEYPOINT_PAIRS):
                # try:
                    if joint_visible[jp[0]] and joint_visible[jp[1]]:
                        currentAxis.plot(pts[jp,0], pts[jp,1],
                                linewidth=2.0, alpha=0.7, color=plt.cm.cool(cm_ind))
                        currentAxis.scatter(pts[jp, 0], pts[jp, 1], s=5)
                # except:
                #     continue

    target_dir ='./labeledImgs/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    plt.axis('off')
    plt.subplots_adjust(top=0.9, bottom=0.01, right=0.9, left=0.01, hspace=0.01, wspace=0.01)
    plt.margins(0, 0)
    # fig.savefig(out_png_path, format='png', transparent=True, dpi=300, pad_inches=0)
    # fig.set_size_inches(1920 / 100, 1080 / 100)
    plt.savefig(target_dir + img_name.replace('/','_'), format='jpg', transparent=True, dpi=300, pad_inches=0)

    plt.show()
    plt.pause(400)
    plt.close()

if __name__=='__main__':
    img_root='/data/COCO/walter1218-datasets-mscoco-1'
    grountruth_file='./custom_coco_train.json'
    vis_one_img(img_root, grountruth_file,vis_num=57000)
    # for i in range(60):
    #     vis_num=i*90
    #     vis_one_img(img_root,grountruth_file,vis_num)
