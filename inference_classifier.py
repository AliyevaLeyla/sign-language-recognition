import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Set a more natural voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Set to a more natural voice (usually index 1)
engine.setProperty('rate', 150)  # Adjust speaking rate

# Function to pronounce the letter
def pronounce_letter(letter):
    engine.say(letter)
    engine.runAndWait()

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture and hand landmarks
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label mapping
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# Variable to store the previous predicted character
previous_character = None

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            data_aux = []
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

            # Predict the character
            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            # Check if the predicted character is different from the previous one
            if predicted_character != previous_character:
                # Pronounce the predicted character
                pronounce_letter(predicted_character)
                # Update the previous character
                previous_character = predicted_character

            # Draw bounding box and predicted character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    # Wait for key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()



