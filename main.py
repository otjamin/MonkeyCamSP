import os
import argparse
import json

from pathlib import Path

import cv2

from dateutil import parser as dateparser

VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "camsp"


def crop(frame, roi):
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def ocr_timestamp(reader, cropped):
    results = reader.readtext(cropped)
    bbox, text, prob = results[0]
    text = text.replace(".", ":")  # Unschöne Korrektur
    timestamp = dateparser.parse(text, fuzzy=True)
    return timestamp.strftime("%Y-%m-%dT%H%M%S")


def select_rois(video_path):
    roi_path = Path(OUTPUT_FOLDER) / "roi.json"
    # Nach Regionen im Ergebnisordner suchen
    if roi_path.exists():
        with open(roi_path, "r") as f:
            rois = json.load(f)
            return rois["timestamp_roi"], rois["door_roi"]

    SELECTION_WINDOW = "roi_select_window"
    cv2.namedWindow(SELECTION_WINDOW, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    cv2.imshow(SELECTION_WINDOW, frame)
    timestamp_roi = cv2.selectROI(SELECTION_WINDOW, frame)
    door_roi = cv2.selectROI(SELECTION_WINDOW, frame)
    cv2.destroyAllWindows()

    # Regionen in JSON speichern
    with open(roi_path, "w") as f:
        json.dump({"timestamp_roi": timestamp_roi, "door_roi": door_roi}, f)

    return timestamp_roi, door_roi


def preprocess_videos(video_folder):
    if not video_folder:
        video_folder = VIDEO_FOLDER

    os.makedirs(Path(OUTPUT_FOLDER) / "out", exist_ok=True)

    video_paths = list(Path(video_folder).glob("*.mp4"))

    timestamp_roi, door_roi = select_rois(video_paths[0])

    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)

    for video in video_paths:
        print(f"Verarbeite {video}...")
        cap = cv2.VideoCapture(video)

        prev_gray = None
        cooldown = 0
        cooldown_frames = 90  # ~3 Sekunden bei 30fps

        # Schwellenwert: 5% der Tür-ROI-Fläche muss sich ändern
        _, _, door_w, door_h = door_roi
        motion_pixel_threshold = int((door_w * door_h) * 0.05)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            door_cropped = crop(frame, door_roi)
            gray = cv2.cvtColor(door_cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is None:
                prev_gray = gray
                continue

            # Hintergrund auch während Cooldown aktualisieren
            if cooldown > 0:
                cooldown -= 1
                prev_gray = gray
                continue

            diff = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            motion_pixels = cv2.countNonZero(thresh)

            if motion_pixels > motion_pixel_threshold:
                cooldown = cooldown_frames

                timestamp_cropped = crop(frame, timestamp_roi)
                timestamp = ocr_timestamp(reader, timestamp_cropped)

                print(f"Tür geöffnet um {timestamp} (motion: {motion_pixels}px)")
                output_path = (
                    Path(OUTPUT_FOLDER) / "out" / f"{video.stem}_{timestamp}.jpg"
                )
                cv2.imwrite(output_path, frame)

            prev_gray = gray


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--videos")
    args = parser.parse_args()

    preprocess_videos(args.videos)
