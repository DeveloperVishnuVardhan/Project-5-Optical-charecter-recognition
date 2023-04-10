"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh

Project-5: CS-5330 -> Spring 2023.

This file contains the code to load the
training and testing data into disk.
"""

import cv2
import torchvision
from helper_functions import load_model
from dataloader import create_dataloaders, get_FashionMnist
import torch
from models import LeNet, tinyVgg

cap = cv2.VideoCapture('/Users/jyothivishnuvardhankolla/Downloads/RPReplay_Final1680647960.MP4')

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307, ), (0.3081,))
])

# Load the models to use.
model1 = LeNet()
model1_path = "Models/base_model.pth"
model1_ = load_model(target_dir=model1_path, model=model1)
train_data1, test_data1, class_names1 = create_dataloaders(32)

model2 = tinyVgg(input_shape=1, hidden_units=10, output_shape=10)
model2_path = "Models/fashion_model.pth"
model2_ = load_model(target_dir=model2_path, model=model2)
train_data2, test_data2, class_names2 = get_FashionMnist(32)

digit_mode = 1
fashion_mode = 0

while True:
    # Capture a frame from the camera.
    ret, frame = cap.read()

    if not ret:
        continue
    
    # Preprocessing frame for predictions.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (28, 28))
    final_img = data_transform(resized_frame)
    final_img = final_img.unsqueeze(0)

    if digit_mode == 1:
        # Perform predictions.
        prediction = model1_(final_img)
        prediction_label = int(torch.argmax(prediction, dim=1))
        label = class_names1[prediction_label]
        print(label)

    if fashion_mode == 1:
        prediction = model2_(final_img)
        prediction_label = int(torch.argmax(prediction, dim=1))
        label = class_names2[prediction_label]
        print(label)

    # Put the live-text on to the video.
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (200, 200)
    font_scale = 5
    color = (0, 255, 0)
    thickness = 5

    cv2.putText(frame, label, location, font, font_scale, color, thickness)
    

    # Display the frame.
    cv2.imshow('Real Time Video', frame)
    k = cv2.waitKey(50)
    if k == ord('m'):
        digit_mode = 1
        fashion_mode = 0

    if k == ord('f'):
        fashion_mode = 1
        digit_mode = 0


cap.release()
cv2.destroyAllWindows()
    