# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import time

from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize

parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# imported related models
# estimator = load_pretrain_model('VGG_origin')
estimator = load_pretrain_model('mobilenet_thin')
action_classifier = load_action_premodel('Action/training/amazon_recognition.h5')

# parameter initialization
realtime_fps = 0.0000
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0
# Read and write video files (tested only for webcam input)
cap = choose_run_mode(args)
video_writer = set_video_writer(cap, write_fps=int(7.0))


# # A txt file that stores joint data for the training process (for training)
#f = open('origin_data.txt', 'a+')

while cv.waitKey(1) < 0:
    has_frame, show = cap.read()
    if has_frame:
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

        height, width = show.shape[:2]
        # Display real-time FPS values
        if (time.time() - start_time) > fps_interval:
            # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
            # Calculate the number of frames in this interval. If the interval is 1 second, it is FPS.
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0  # Clear the number of frames
            start_time = time.time()
        fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
        cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show the number of people detected
        num_label = "Human: {0}".format(len(humans))
        cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show current run time and total frames
        if frame_count == 1:
            run_timer = time.time()
        run_time = time.time() - run_timer
        time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
        cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv.imshow('Action Recognition based on OpenPose', show)
        video_writer.write(show)

        # # Collect data for training process (for training)
        # joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
        # f.write(' '.join(joints_norm_per_frame))
        # f.write('\n')

video_writer.release()
cap.release()
# f.close()
