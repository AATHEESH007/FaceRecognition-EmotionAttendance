import csv
import os
import numpy as np
import cv2

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

yaar_athu = cv2.face.LBPHFaceRecognizer_create()

cam = cv2.VideoCapture(0)

id = int(input("Enter your id: "))
name = input("Enter your name: ")


count = 0

faces = []
face_id = []

print("Starting face data collection")

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_face = detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in detect_face:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        
        faces.append(face_img)
        face_id.append(id)
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, f"Sample : {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
    cv2.imshow("Capturing ...", frame)
    
    key = cv2.waitKey(10)
        
    if key == 27 or count >= 30:
        break
        
print("collection complete")
print("Training the model...")
yaar_athu.train(faces, np.array(face_id))


os.makedirs("model", exist_ok=True)
yaar_athu.save("model/face_model.yml")

cam.release()
cv2.destroyAllWindows() 

def save_names(id, name):
    file_exists = os.path.isfile("students.csv")

    with open("students.csv", mode="a", newline="") as file:
        writer = csv.writer(file)

        # write header only once
        if not file_exists:
            writer.writerow(["id", "name"])

        writer.writerow([id, name])

save_names(id, name)
print("Model trained and saved successfully.")  