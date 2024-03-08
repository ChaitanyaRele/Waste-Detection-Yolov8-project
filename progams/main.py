from ultralytics import YOLO
import os
import math
import torch



def main():
    print(torch.cuda.is_available())

    path = "D:\\Nanomod\\TrashClass-2"
    try:
        contents = os.listdir(path)
        print(contents)
    except OSError as e:
        print(f"Error: {e}")

    model = YOLO("yolov8n.yaml")
    results = model.train(data = "D:\\Nanomod\\TrashClass-2\\data.yaml",epochs = 30, save = True, save_period = 2)

if __name__ == '__main__':
    main()
