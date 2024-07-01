import os
import pandas as pd
import cv2
from datetime import datetime
from deepface import DeepFace
from PIL import Image, ImageTk
import tkinter as tk

TRAIN_DIR = "./train_images"
ATTENDANCE_FILE = "attendance.csv"


# Função para carregar imagens de treino e codificá-las
def load_train_images(TRAIN_DIR):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(TRAIN_DIR):
        person_dir = os.path.join(TRAIN_DIR, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(person_dir, filename)
                    known_face_names.append(person_name)
                    known_face_encodings.append(image_path)

    return known_face_encodings, known_face_names

# Carregar imagens de treino
known_face_encodings, known_face_names = load_train_images(TRAIN_DIR)

# Função para inicializar o arquivo de presença
def initialize_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE) or os.stat(ATTENDANCE_FILE).st_size == 0:
        df = pd.DataFrame(columns=['Name', 'Timestamp'])
        df.to_csv(ATTENDANCE_FILE, index=False)

# Função para marcar presença
def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    attendance_data = {'Name': name, 'Timestamp': dt_string}
    
    df = pd.read_csv(ATTENDANCE_FILE)
    if not ((df['Name'] == name) & (df['Timestamp'].str.contains(dt_string.split(' ')[0]))).any():
        df = pd.concat([df, pd.DataFrame([attendance_data])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE,index=False)

# Função para verificar a vivacidade
def is_live_face(face_image):
    # Use DeepFace to analyze the face
    analysis = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
    # If DeepFace returns a face, consider it live (this is a simplified approach)
    return len(analysis) > 0

# Função para reconhecimento facial
def recognize_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = DeepFace.extract_faces(img_path=rgb_frame, enforce_detection=False)

    for face in faces:
        face_image = face['face']

        if not is_live_face(face_image):
            continue

        name = "Unknown"
        for idx, known_face in enumerate(known_face_encodings):
            result = DeepFace.verify(img1_path=face_image, img2_path=known_face, enforce_detection=False)
            if result["verified"]:
                name = known_face_names[idx]
                break

        if name != "Unknown":
            mark_attendance(name)

        facial_area = face['facial_area']
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 35), (x + w, y), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)

    return frame

# Função para iniciar a webcam e reconhecimento facial
def start_recognition():
    cap = cv2.VideoCapture(0)

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = recognize_face(frame)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
        lmain.after(10, update_frame)

    update_frame()

# Inicializar o arquivo de presença
initialize_attendance_file()

# Configuração da interface gráfica
root = tk.Tk()
root.title("Reconhecimento Facial")

recognize_button = tk.Button(root, text="Iniciar Reconhecimento Facial", command=start_recognition)
recognize_button.pack(pady=20)

lmain = tk.Label(root)
lmain.pack()

root.mainloop()

