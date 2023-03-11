import pandas as pd
import cv2, os
import mediapipe as mp


# names of the landmarks (joints) that are to be collected
results = {
    'right_shoulder_x': [], #11
    'right_shoulder_y': [], #11
    'left_shoulder_x': [], # 12
    'left_shoulder_y': [], #12
    'right_elbow_x': [], #13
    'right_elbow_y': [], #13
    'left_elbow_x': [], #14
    'left_elbow_y': [], #14
    'right_wrist_x': [], #15
    'right_wrist_y': [], #15
    'left_wrist_x': [], #16
    'left_wrist_y': [], #16
    'right_hip_x' : [], #23
    'right_hip_y' : [], #23
    'left_hip_x' : [], #24
    'left_hip_y' : [], #24
    'right_knee_x' : [], #25
    'right_knee_y' : [], #25
    'left_knee_x' : [], #26
    'left_knee_y' : [], #26
    'right_ankle_x' : [], #27
    'right_ankle_y' : [], #27
    'left_ankle_x' : [], #28
    'left_ankle_y' : [], #28
    #'shot_name' : []
              }


def get_landmarks(folder_path):
    file_names = os.listdir(folder_path)
    for file in file_names:
        image = cv2.imread(folder_path + '/' + file)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_joints = pose.process(image_rgb)
        if results_joints.pose_landmarks is not None:
            results['right_shoulder_x'].append(results_joints.pose_landmarks.landmark[11].x)
            results['right_shoulder_y'].append(results_joints.pose_landmarks.landmark[11].y)
            results['left_shoulder_x'].append(results_joints.pose_landmarks.landmark[12].x)
            results['left_shoulder_y'].append(results_joints.pose_landmarks.landmark[12].y)
            results['right_elbow_x'].append(results_joints.pose_landmarks.landmark[13].x)
            results['right_elbow_y'].append(results_joints.pose_landmarks.landmark[13].y)
            results['left_elbow_x'].append(results_joints.pose_landmarks.landmark[14].x)
            results['left_elbow_y'].append(results_joints.pose_landmarks.landmark[14].y)
            results['right_wrist_x'].append(results_joints.pose_landmarks.landmark[15].x)
            results['right_wrist_y'].append(results_joints.pose_landmarks.landmark[15].y)
            results['left_wrist_x'].append(results_joints.pose_landmarks.landmark[16].x)
            results['left_wrist_y'].append(results_joints.pose_landmarks.landmark[16].y)
            results['right_hip_x'].append(results_joints.pose_landmarks.landmark[23].x)
            results['right_hip_y'].append(results_joints.pose_landmarks.landmark[23].y)
            results['left_hip_x'].append(results_joints.pose_landmarks.landmark[24].x)
            results['left_hip_y'].append(results_joints.pose_landmarks.landmark[24].y)
            results['right_knee_x'].append(results_joints.pose_landmarks.landmark[25].x)
            results['right_knee_y'].append(results_joints.pose_landmarks.landmark[25].y)
            results['left_knee_x'].append(results_joints.pose_landmarks.landmark[26].x)
            results['left_knee_y'].append(results_joints.pose_landmarks.landmark[26].y)
            results['right_ankle_x'].append(results_joints.pose_landmarks.landmark[27].x)
            results['right_ankle_y'].append(results_joints.pose_landmarks.landmark[27].y)
            results['left_ankle_x'].append(results_joints.pose_landmarks.landmark[28].x)
            results['left_ankle_y'].append(results_joints.pose_landmarks.landmark[28].y)
            # assigning value to each kind of shot so it can be used to train classification ML models
            if 'Drive' in folder_path:
                results['shot_name'].append(1)
            elif 'Pullshot' in folder_path:
                results['shot_name'].append(2)
            elif 'Sweep' in folder_path:
                results['shot_name'].append(3)
            elif 'Legglance' in folder_path:
                results['shot_name'].append(4)

# names of the folders to get images from
folder_paths = ['images']
for folder_path in folder_paths:
    get_landmarks(folder_path)
