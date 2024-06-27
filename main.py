import os
import pandas as pd
from datetime import datetime

TRAIN_DIR = "./train_images"
ATTENDANCE_FILE = "attendance.csv"


# Função para carregar imagens de treino e codificá-las
def load_train_images(TRAIN_DIR):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(TRAIN_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(TRAIN_DIR, filename)
            known_face_names.append(os.path.splitext(filename)[0])
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