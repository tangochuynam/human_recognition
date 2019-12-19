# -*- coding: UTF-8 -*-
import cv2 as cv
import os
import argparse
import numpy as np
import pandas as pd
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize

parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument( '-img', '--image', required="True", help='Path to image folder.')
args = parser.parse_args()

# imported related models
estimator = load_pretrain_model('VGG_origin')
action_classifier = load_action_premodel('Action/framewise_recognition.h5')

# parameter initialization
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0
folder_path = args.image

# create df for saving joints 
columns = ["nose_x", "nose_y", "neck_x", "neck_y", "Rshoulder_x", "Rshoulder_y", "Relbow_x", 
"Relbow_y", "Rwrist_x", "RWrist_y", "LShoulder_x", "LShoulder_y", "LElbow_x", "LElbow_y", 
"LWrist_x", "LWrist_y", "RHip_x", "RHip_y", "RKnee_x", "RKnee_y", "RAnkle_x", "RAnkle_y", 
"LHip_x", "LHip_y", "LKnee_x", "LKnee_y", "LAnkle_x", "LAnkle_y", "REye_x", "REye_y", 
"LEye_x", "LEye_y", "REar_x", "REar_y", "LEar_x", "LEar_y", "class"]
df = pd.DataFrame(columns=columns)

for f_name in os.listdir(folder_path): 
    sub_f = folder_path + "/" + f_name
    # folder_out = "test_out" + "/" + f_name
    print("f_name: " + f_name)
    # if not os.path.isdir(folder_out):
    #     os.mkdir(folder_out)
    for img in os.listdir(sub_f):
        print("image name: " + img)
        show = cv.imread(sub_f + "/" + img)
        fps_count += 1
        frame_count += 1

        # pose estimation
        humans = estimator.inference(show)
        
        # print(len(humans))
        # print(humans[0].uidx_list)
        # print(humans[0].body_parts)

        # get pose info
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter
        # recognize the action framewise
        show = framewise_recognize(pose, action_classifier)

        # height, width = show.shape[:2]
        # # Display real-time FPS values
        # if (time.time() - start_time) > fps_interval:
        #     # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
        #     # Calculate the number of frames in this interval. If the interval is 1 second, it is FPS.
        #     realtime_fps = fps_count / (time.time() - start_time)
        #     fps_count = 0  # Clear the number of frames
        #     start_time = time.time()
        # fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
        # cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # # Show the number of people detected
        # num_label = "Human: {0}".format(len(humans))
        # cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # # Show current run time and total frames
        # if frame_count == 1:
        #     run_timer = time.time()
        # run_time = time.time() - run_timer
        # time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
        # cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # cv.imshow('Action Recognition based on OpenPose', show)
        # img_out = img.split(".")[0] + "_out_" + ".png" 
        # cv.imwrite(folder_out + "/" + img, show)
        # video_writer.write(show)

        # # Collect data for training process (for training)
        joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
        # print("length of joints frames: " + str(len(joints_norm_per_frame)))
        # only select joints_norm_per_frame with 1 human 
        if len(joints_norm_per_frame) == 36: 
            row = np.append(joints_norm_per_frame, f_name) 
            series = pd.Series(dict(zip(df.columns, row)))
            df = df.append(series, ignore_index=True)

# saving df to csv
df.to_csv("Action/training/human_keypoint.csv", index=False)

