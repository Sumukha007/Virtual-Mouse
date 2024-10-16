import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            index_finger = landmarks[8]
            thumb = landmarks[4]
            middle_finger = landmarks[12]

            # Get screen coordinates
            index_x = int(index_finger.x * frame_width)
            index_y = int(index_finger.y * frame_height)
            thumb_x = int(thumb.x * frame_width)
            thumb_y = int(thumb.y * frame_height)
            middle_x = int(middle_finger.x * frame_width)
            middle_y = int(middle_finger.y * frame_height)

            # Draw circles on fingers
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), -1)
            cv2.circle(frame, (middle_x, middle_y), 10, (255, 0, 0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)

            # Detect gestures
            if abs(index_y - thumb_y) < 20:  # Left click
                pyautogui.click()
                pyautogui.sleep(1)
            elif abs(middle_y - thumb_y) < 20:  # Right click
                pyautogui.click(button='right')
                pyautogui.sleep(1)
            elif abs(index_y - thumb_y) < 20 and abs(middle_y - thumb_y) < 20:  # Selection
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()

            # Move mouse
            pyautogui.moveTo(screen_width/frame_width*index_x, screen_height/frame_height*index_y)

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
