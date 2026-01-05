import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter
import warnings
from TTS import TextToSpeech

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

from pynput.keyboard import Key, Controller
keyboard = Controller()
gameMode = False
gameSettings = False

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.8)

labels_dict = {
    0: 'A',  1: 'B',  2: 'C',  3: 'D',  4: 'E',
    5: 'F',  6: 'G',  7: 'H',  8: 'I',  9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26:"SPACE", 27:"MODE"
}

letters_buffer = []
output_text = ""
last_printed = ""
last_time = time.time()
confirm_time_seconds = 3  # cada cuántos segundos confirmar la letra
modeText = "Write Mode"

while True and not gameMode:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            prediction = model.predict([np.asarray(data_aux)])
        except ValueError:
            pass
        
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    current_time = time.time()
    if current_time - last_time >= confirm_time_seconds:
        if letters_buffer:
            most_common_letter, count = Counter(letters_buffer).most_common(1)[0]
            if most_common_letter != last_printed:
                print(f"Letra confirmada: {most_common_letter}")
                if most_common_letter=="SPACE":
                    output_text += " "
                elif most_common_letter=="MODE":
                    pass
                else:
                    output_text += most_common_letter
                last_printed = most_common_letter

                if (most_common_letter == "MODE"):
                    gameMode = True
                    modeText = "Game Mode"

        letters_buffer.clear()
        last_time = current_time

    if predicted_character is not None:
        letters_buffer.append(predicted_character)
    else:
        # Cuando no detectamos letra, vaciamos buffer y reseteamos para permitir repetir letras luego
        letters_buffer.clear()
        last_printed = ""

    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        break

    cv2.putText(frame, modeText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)    
    cv2.imshow('frame', frame)


############################################################
###     Entra en este bucle si se activa el gameMode     ###
############################################################

while True and gameMode:
    if (not gameSettings):
        time.sleep(1)
        gameSettings = True
        
        labels_dict = {
            0: 'A',  1: 'B',  2: 'C',  3: 'D',  4: 'E',
            5: 'F',  6: 'G',  7: 'H',  8: 'I',  9: 'J',
            10: 'K', 11: 'J', 12: 'M', 13: 'N', 14: 'O',
            15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
            20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
            25: 'Z', 26:"SPACE", 27:"MODE"
        }

        confirm_time_seconds = 0.01  # input mas rapido para juegos

        print("GAMEMODE ON")

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            prediction = model.predict([np.asarray(data_aux)])
        except ValueError:
            pass
        
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    current_time = time.time()
    if current_time - last_time >= confirm_time_seconds:
        if letters_buffer:
            if (most_common_letter != "MODE"):
                keyboard.release(most_common_letter)
            
            most_common_letter, count = Counter(letters_buffer).most_common(1)[0]

            if (most_common_letter == "SPACE"):
                most_common_letter = Key.space
            # if (most_common_letter == "DOWN"):
            #     most_common_letter = Key.down

            if most_common_letter != last_printed:
                print(f"Letra confirmada: {most_common_letter}")
                last_printed = most_common_letter

                if (most_common_letter == "MODE"):
                    gameMode = False
                    break
                
                keyboard.press(most_common_letter)

        letters_buffer.clear()
        last_time = current_time

    if predicted_character is not None:
        letters_buffer.append(predicted_character)
    else:
        # Cuando no detectamos letra, vaciamos buffer y reseteamos para permitir repetir letras luego
        letters_buffer.clear()
        last_printed = ""

    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        break
    
    cv2.putText(frame, modeText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()

print("\nTexto final reconocido:", output_text)
output_text = output_text*3
TextToSpeech(output_text)