import os
import cv2
from drowsy_detection import FrameHandler
from object_detection import yolo_detection
import numpy as np
import copy
import sys
import argparse


config_drowsy_detection = {
    "EAR_THRESH": 0.2,
    "WAIT_TIME": 1,
    "TIME_ONE_FRAME": 0.1
}


def run_video_processing(file_path):
    handler = FrameHandler()
    cap = cv2.VideoCapture(file_path)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_one_frame = 1/fps
    config_drowsy_detection["TIME_ONE_FRAME"] = time_one_frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/out_{}'.format(os.path.basename(file_path)), fourcc, 20.0, (width,height))
    while True:
        ret, frame = cap.read()
        if not type(frame) == np.ndarray:
            break
        draw_frame = frame.copy()
        draw_frame = handler.process(frame, draw_frame, config_drowsy_detection)
        yolo_detection(frame, draw_frame)
        out.write(draw_frame)
        c = cv2.waitKey(1)
    cap.release()
    out.release()


def run_image_processing(file_path):
    handler = FrameHandler()
    detect_img = cv2.imread(file_path)
    draw_img = detect_img.copy()
    yolo_detection(detect_img, draw_img)
    draw_img = handler.process(detect_img, draw_img, config_drowsy_detection)
    cv2.imwrite('output/out_{}'.format(os.path.basename(file_path)), draw_img)


def check_type_files(file_path, data_type):
    if data_type == 'image':
        return file_path.lower().endswith(('.png', '.jpg'))
    else:
        return file_path.lower().endswith('.mp4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driver computer vision')
    parser.add_argument('--data_type', type=str, choices=['image', 'video'])
    parser.add_argument('--file_path', type=str)   
    args = parser.parse_args()
    parser.print_help()
    if not check_type_files(args.file_path, args.data_type):
        print("Does not match the file extension. Video accepts type '.mp4'. Image accepts types: '.png', '.jpg'")
        exit()
    if args.data_type == 'video':
        run_video_processing(args.file_path)
    else:
        run_image_processing(args.file_path)
