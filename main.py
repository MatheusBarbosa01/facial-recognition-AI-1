import os

TRAIN_DIR = "./train_images"

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