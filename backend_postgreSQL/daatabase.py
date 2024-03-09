import customtkinter as ctk
from PIL import Image, ImageTk
from multiprocessing import Process
import cv2
from tkinter import filedialog
from ultralytics import YOLO
import cvzone
import math

import time
import psycopg2

def start():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("D:\\Nanomod\\runs\\detect\\train9\\weights\\best.pt")

    classNames = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]
    myColor = (0, 0, 255)

    last_added = {class_name: 0 for class_name in classNames}

    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="universe",
        host="localhost",
        port="5432"
    )


    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS waste (
            id SERIAL PRIMARY KEY,
            class_name VARCHAR(255),
            box_xmin INTEGER,
            box_ymin INTEGER,
            box_xmax INTEGER,
            box_ymax INTEGER,
            confidence FLOAT
        )
    """)

    # Commit the transaction
    conn.commit()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame from webcam. Exiting ...")
            break

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1


                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                currentClass = classNames[cls]
                print(currentClass)
                if conf > 0.65:  # Changed this line
                    if currentClass == 'BIODEGRADABLE':
                        myColor = (0, 255, 0)
                    elif currentClass == 'CARDBOARD':
                        myColor = (0, 0, 255)
                    elif currentClass == 'GLASS':
                        myColor = (255, 0, 0)
                    elif currentClass == 'METAL':
                        myColor = (255, 255, 0)
                    elif currentClass == 'PAPER':
                        myColor = (0, 255, 255)
                    elif currentClass == 'PLASTIC':
                        myColor = (255, 0, 255)

                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                     (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(0, 0, 0), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 1)


                    if time.time() - last_added[currentClass] >= 7.0:
                        # Update the last added time for this class
                        last_added[currentClass] = time.time()


                        cur.execute("INSERT INTO waste (class_name, box_xmin, box_ymin, box_xmax, box_ymax,confidence) VALUES (%s, %s, %s, %s, %s,%s)",
                                    (currentClass, x1, y1, x2, y2,confidence))


                        conn.commit()

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    conn.close()

def predict():
    classNames = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]

    # Open file dialog to select the image file
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if not filepath:
        return

    # Load the image
    img = cv2.imread(filepath)

    # Resize the image to 1280x720
    img = cv2.resize(img, (1280, 720))

    # Load YOLOv8
    model = YOLO("D:\\Nanomod\\runs\\detect\\train9\\weights\\best.pt")
    names = model.names

    # Predict the classes
    results = model(img, show=True)

    # Connect to your postgres DB
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="universe",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS waste (
            id SERIAL PRIMARY KEY,
            class_name VARCHAR(255),
            box_xmin INTEGER,
            box_ymin INTEGER,
            box_xmax INTEGER,
            box_ymax INTEGER,
            confidence FLOAT
        )
    """)

    conn.commit()

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for xyxy, c, confidence in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1 = (int(xyxy[0]),int(xyxy[1]))
            x2, y2 = (int(xyxy[2]),int(xyxy[3]))
            class_name = names[int(c)]
            conf = float(confidence)
            print(class_name)
            print(conf)
            print((x1,y1),(x2,y2))
            cv2.rectangle(img,(x1, y1), (x2, y2),(0,255,0))

            cur.execute("""
                INSERT INTO waste (class_name, box_xmin, box_ymin, box_xmax, box_ymax,confidence )
                VALUES (%s, %s, %s, %s, %s,%s)
            """, (class_name, x1, y1, x2, y2,conf))

        conn.commit()
    conn.close()

    return img


if __name__ == "__main__":

    root = ctk.CTk()
    root.title("Waste Classifier")
    root.geometry("800x600")


    title = ctk.CTkLabel(root, text="Waste Classifier", font=("Helvetica", 32))
    title.pack()


    image = Image.open("dustbin.jpg")
    photo = ImageTk.PhotoImage(image)
    label = ctk.CTkLabel(root, image=photo)
    label.image = photo
    label.pack()


    start_button = ctk.CTkButton(root, text="Start", command=lambda: Process(target=start).start())
    start_button.pack()


    predict_button = ctk.CTkButton(root, text="Predict Image", command=predict)
    predict_button.pack()

    root.mainloop()
