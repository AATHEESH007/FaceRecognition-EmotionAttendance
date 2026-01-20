import csv
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from datetime import datetime


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model/face_model.yml")


emo_model = load_model("emotion_model.h5", compile=False)  # emo model

model = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)  # model

webcam = cv2.VideoCapture(0)  # webcam

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

students = {}
with open("students.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        students[int(row["id"])] = row["name"]

attendance_marked = set()
emotion_buffer = {}


def dominant_emotion(emotion_list):
    return max(set(emotion_list), key=emotion_list.count)


while True:
    working, frames = webcam.read()
    if not working:
        break

    bnw = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)  # convert to black and white

    face = model.detectMultiScale(
        bnw, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )  # detect faces

    for (x, y, w, h) in face:  # draw rectangle around face
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_for_id = bnw[y:y + h, x:x + w]
        face_for_id = cv2.resize(face_for_id, (200, 200))

        predicted_id, confidence = recognizer.predict(face_for_id)

        if confidence < 60 and predicted_id in students:
            name = students[predicted_id]
        else:
            name = "theriyala"

        face_image = bnw[y:y + h, x:x + w]  # croped face
        face_image = cv2.resize(face_image, (64, 64))
        face_image = face_image / 255.0
        face_image = np.reshape(face_image, (1, 64, 64, 1))

        emo = emo_model.predict(face_image, verbose=0)
        emotion = emotions[np.argmax(emo)]

        if name != "theriyala":
            if name not in emotion_buffer:
                emotion_buffer[name] = []

            emotion_buffer[name].append(emotion)

            if len(emotion_buffer[name]) > 20:
                emotion_buffer[name].pop(0)

            if len(emotion_buffer[name]) >= 10:
                dom_emo = dominant_emotion(emotion_buffer[name])

                cv2.putText(
                    frames,
                    f"{name} - {dom_emo}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                if name not in attendance_marked:
                    now = datetime.now()

                    with open("attendance.csv", "a", newline="") as file:
                        writer = csv.writer(file)
                        if file.tell() == 0:
                            writer.writerow(["Name", "Date", "Time", "Emotion"])
                        writer.writerow([
                            name,
                            now.strftime("%d-%m-%Y"),
                            now.strftime("%H:%M:%S"),
                            dom_emo
                        ])

                    attendance_marked.add(name)
        else:
            cv2.putText(
                frames,
                "theriyala",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

    cv2.imshow("muga raasi", frames)  # display video

    if cv2.waitKey(10) == 27:  # ESC
        break

webcam.release()
cv2.destroyAllWindows()
