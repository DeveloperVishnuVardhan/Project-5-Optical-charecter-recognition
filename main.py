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
from dataloader import create_dataloaders
import torch
from models import LeNet

cap = cv2.VideoCapture('/Users/jyothivishnuvardhankolla/Downloads/RPReplay_Final1680647960.MP4')

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307, ), (0.3081,))
])

# Load the model to use.
model = LeNet()
model_path = "Models/base_model.pth"
model = load_model(target_dir=model_path, model=model)
train_data, test_data, class_names = create_dataloaders(32)

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

    # Perform predictions.
    prediction = model(final_img)
    prediction_label = int(torch.argmax(prediction, dim=1))
    print(class_names[prediction_label])

    # Put the live-text on to the video.
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (200, 200)
    font_scale = 5
    color = (0, 255, 0)
    thickness = 5

    cv2.putText(frame, class_names[prediction_label], location, font, font_scale, color, thickness)
    

    # Display the frame.
    cv2.imshow('Real Time Video', frame)
    k = cv2.waitKey(50)

cap.release()
cv2.destroyAllWindows()
    