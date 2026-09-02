from pathlib import Path
import urllib.request
import math

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from pycaw.pycaw import AudioUtilities


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def draw_landmarks(image, landmarks):
    for hand_landmarks in landmarks:
        points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 0), -1)

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]
        for start, end in connections:
            cv2.line(image, points[start], points[end], (0, 0, 0), 2)


def draw_volume_bar(image, scalar):
    bar_x, bar_y = 20, 60
    bar_width, bar_height = 250, 25
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), 2)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(bar_width * scalar), bar_y + bar_height), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"Volume: {int(scalar * 100)}%",
        (bar_x, bar_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )


def main():
    speakers = AudioUtilities.GetSpeakers()
    volume = speakers.EndpointVolume

    model_path = ensure_model()
    options = hand_landmarker.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
        num_hands=1,
    )

    my_hands = hand_landmarker.HandLandmarker.create_from_options(options)
    webcam = cv2.VideoCapture(0)

    while True:
        success, image = webcam.read()
        if not success:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        output = my_hands.detect(mp_image)

        if output.hand_landmarks:
            draw_landmarks(image, output.hand_landmarks)
            hand = output.hand_landmarks[0]
            thumb = hand[4]
            index = hand[8]

            x1 = int(thumb.x * image.shape[1])
            y1 = int(thumb.y * image.shape[0])
            x2 = int(index.x * image.shape[1])
            y2 = int(index.y * image.shape[0])

            # draw connecting line between thumb and index as light pink (BGR)
            cv2.line(image, (x1, y1), (x2, y2), (203, 192, 255), 2)
            # draw the landmark points as part of the black skeletal style
            cv2.circle(image, (x1, y1), 8, (0, 0, 0), -1)
            cv2.circle(image, (x2, y2), 8, (0, 0, 0), -1)

            distance = math.hypot(x2 - x1, y2 - y1)
            min_distance = 25
            max_distance = 200
            distance = max(min_distance, min(distance, max_distance))
            volume_scalar = (distance - min_distance) / (max_distance - min_distance)
            volume_scalar = max(0.0, min(volume_scalar, 1.0))
            volume.SetMasterVolumeLevelScalar(volume_scalar, None)

            draw_volume_bar(image, volume_scalar)
            cv2.putText(
                image,
                f"Thumb-index dist: {int(distance)}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
        else:
            cv2.putText(
                image,
                "Show thumb and index finger to change volume",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        cv2.imshow("Hand volume control", image)
        key = cv2.waitKey(10)
        if key == 27:
            break

    webcam.release()
    my_hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

 