import os
import sys
import subprocess
from pathlib import Path


def main():
    print("Starting MonkeyCamSP...")
    # Find the path to this script file
    script_path = os.path.abspath(__file__)
    # Run streamlit pointing to this file
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])
    sys.exit(0)


# If this is run via the generated entrypoint script, streamlit hasn't been started yet.
# We check if we are already in a streamlit run.
if "streamlit" not in sys.modules:
    pass  # Wait for main() to be called
elif os.path.basename(sys.argv[0]) == "streamlit":
    pass  # We are being run by streamlit

# The rest of the code is executed when streamlit runs this file
import streamlit as st
import cv2
import json
import csv
from dateutil import parser as dateparser

# Constants
VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "camsp"
CSV_PATH = Path(OUTPUT_FOLDER) / "ergebnisse.csv"
NAMES_PATH = Path("names.txt")

# Ensure output folder exists
os.makedirs(Path(OUTPUT_FOLDER) / "out", exist_ok=True)


# ... backend logic from main.py ...
def crop(frame, roi):
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def ocr_timestamp(reader, cropped):
    results = reader.readtext(cropped)
    if not results:
        return "Unknown"
    bbox, text, prob = results[0]
    text = text.replace(".", ":")  # Unschöne Korrektur
    try:
        timestamp = dateparser.parse(text, fuzzy=True)
        return timestamp.strftime("%Y-%m-%dT%H%M%S")
    except Exception:
        return text


def draw_instruction_banner(frame, text):
    """Draws a semi-transparent black banner with white text at the top of the frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    banner_height = 60

    # Draw black rectangle
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
    # Blend with original frame (50% opacity)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (255, 255, 255)  # White

    # Calculate text position to center it roughly
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (banner_height + text_size[1]) // 2

    cv2.putText(
        frame,
        text,
        (max(10, text_x), text_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def select_rois(video_path):
    roi_path = Path(OUTPUT_FOLDER) / "roi.json"
    if roi_path.exists():
        with open(roi_path, "r") as f:
            rois = json.load(f)
            return rois["timestamp_roi"], rois["door_roi"]

    cap = cv2.VideoCapture(video_path)
    ret, original_frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame from video: {video_path}")

    # --- STEP 1: Select Timestamp ---
    WINDOW_TITLE_1 = "Schritt 1: Zeitstempel (Timestamp)"
    cv2.namedWindow(WINDOW_TITLE_1, cv2.WINDOW_NORMAL)

    frame_step1 = original_frame.copy()
    frame_step1 = draw_instruction_banner(
        frame_step1, "SCHRITT 1: ZEITSTEMPEL (Uhrzeit) markieren und ENTER druecken"
    )

    # showCrosshair=True, fromCenter=False
    timestamp_roi = cv2.selectROI(WINDOW_TITLE_1, frame_step1, True, False)
    cv2.destroyWindow(WINDOW_TITLE_1)

    # --- STEP 2: Select Door ---
    WINDOW_TITLE_2 = "Schritt 2: Tuer (Door)"
    cv2.namedWindow(WINDOW_TITLE_2, cv2.WINDOW_NORMAL)

    frame_step2 = original_frame.copy()
    frame_step2 = draw_instruction_banner(
        frame_step2, "SCHRITT 2: TUER (Eingang) markieren und ENTER druecken"
    )

    # Draw the already selected timestamp ROI as visual feedback
    if timestamp_roi != (0, 0, 0, 0):
        x, y, w, h = timestamp_roi
        cv2.rectangle(frame_step2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame_step2,
            "Zeitstempel",
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    door_roi = cv2.selectROI(WINDOW_TITLE_2, frame_step2, True, False)
    cv2.destroyWindow(WINDOW_TITLE_2)

    with open(roi_path, "w") as f:
        json.dump({"timestamp_roi": timestamp_roi, "door_roi": door_roi}, f)

    return timestamp_roi, door_roi


def preprocess_videos(video_folder, progress_bar, status_text):
    if not video_folder:
        video_folder = VIDEO_FOLDER

    os.makedirs(Path(OUTPUT_FOLDER) / "out", exist_ok=True)
    video_paths = list(Path(video_folder).glob("*.mp4"))

    if not video_paths:
        status_text.error(f"Keine .mp4 Videos im Ordner {video_folder} gefunden.")
        return False

    status_text.info("Bitte ROI im OpenCV Fenster auswählen...")
    timestamp_roi, door_roi = select_rois(str(video_paths[0]))

    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)

    total_videos = len(video_paths)
    for i, video in enumerate(video_paths):
        status_text.info(f"Verarbeite Video {i + 1}/{total_videos}: {video.name}...")
        cap = cv2.VideoCapture(str(video))

        # Calculate total frames for inner progress
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        prev_gray = None
        cooldown = 0
        cooldown_frames = 90  # ~3 Sekunden bei 30fps

        _, _, door_w, door_h = door_roi
        motion_pixel_threshold = int((door_w * door_h) * 0.05)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                # Update progress roughly every second
                progress = (i / total_videos) + (frame_count / total_frames) * (
                    1 / total_videos
                )
                # Ensure progress stays between 0 and 1
                progress = max(0.0, min(1.0, progress))
                progress_bar.progress(progress)

            door_cropped = crop(frame, door_roi)
            gray = cv2.cvtColor(door_cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is None:
                prev_gray = gray
                continue

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

                output_path = (
                    Path(OUTPUT_FOLDER) / "out" / f"{video.stem}_{timestamp}.jpg"
                )
                cv2.imwrite(str(output_path), frame)

            prev_gray = gray

    progress_bar.progress(1.0)
    status_text.success("Verarbeitung abgeschlossen!")
    return True


# ... UI logic ...
def extract_timestamp(filename: str) -> str:
    stem = Path(filename).stem
    return stem.rsplit("_", 1)[-1]


def save_csv():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "action", "name"])
        writer.writerows(st.session_state.results)


def load_names():
    if NAMES_PATH.exists():
        return [n.strip() for n in NAMES_PATH.read_text().splitlines() if n.strip()]
    return []


def save_names(names):
    NAMES_PATH.write_text("\n".join(names))


# Execution logic
# If running through python directly or via our entrypoint script
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if not get_script_run_ctx():
        main()
except ImportError:
    pass

# --- Streamlit App UI ---
st.set_page_config(page_title="MonkeyCamSP", layout="wide")

import tkinter as tk
from tkinter import filedialog


def select_folder():
    """Opens a native dialog to select a folder and returns the path."""
    root = tk.Tk()
    root.withdraw()
    # Make it stay on top
    root.wm_attributes("-topmost", 1)
    folder_path = filedialog.askdirectory()
    root.destroy()
    return folder_path


st.title("MonkeyCamSP - Zuweisungstool")

# Initialize Session State
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "setup"  # "setup", "processing", "triage"
if "index" not in st.session_state:
    st.session_state.index = 0
if "results" not in st.session_state:
    st.session_state.results = []
if "names" not in st.session_state:
    st.session_state.names = load_names()
if "video_folder_input" not in st.session_state:
    st.session_state.video_folder_input = VIDEO_FOLDER

# Setup Mode
if st.session_state.app_mode == "setup":
    st.header("1. Videos Verarbeiten")
    st.write("Wählen Sie den Ordner mit den Videos und starten Sie die Verarbeitung.")

    # Check if a folder was selected before rendering the text_input to avoid the "cannot be modified after instantiation" error
    if "folder_to_set" in st.session_state:
        st.session_state.video_folder_input = st.session_state.folder_to_set
        del st.session_state.folder_to_set

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        # Use session_state key directly to bind the widget to the state
        st.text_input("Ordner mit .mp4 Videos", key="video_folder_input")
    with col_btn:
        # Align button with text input
        st.write("")
        st.write("")
        if st.button("📁 Ordner wählen"):
            selected = select_folder()
            if selected:
                # Save it to a temporary variable and rerun so we can set it BEFORE rendering the widget next cycle
                st.session_state.folder_to_set = selected
                st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Videos verarbeiten", type="primary"):
            st.session_state.app_mode = "processing"
            st.session_state.video_folder = st.session_state.video_folder_input
            st.rerun()

    with col2:
        if st.button("Überspringen (Direkt zur Zuweisung)"):
            st.session_state.app_mode = "triage"
            st.rerun()

# Processing Mode
elif st.session_state.app_mode == "processing":
    st.header("1. Videos Verarbeiten")

    # Keep track of processing state
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False

    if st.session_state.processing_done:
        st.progress(1.0)
        st.success("Verarbeitung abgeschlossen!")
        if st.button("Weiter zur Zuweisung", type="primary", key="continue_btn"):
            st.session_state.app_mode = "triage"
            st.session_state.processing_done = False
            st.rerun()
        if st.button("Zurück"):
            st.session_state.app_mode = "setup"
            st.session_state.processing_done = False
            st.rerun()
    else:
        # Needs to be rendered before blocking call
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        st.info(
            "Hinweis: Falls kein roi.json existiert, öffnet sich ein Fenster. Bitte ROI aufziehen und mit SPACE oder ENTER bestätigen. Zuerst Timestamp, dann Tür."
        )

        if st.button("Zurück"):
            st.session_state.app_mode = "setup"
            st.rerun()

        # Create a placeholder for the start button
        start_placeholder = st.empty()
        if start_placeholder.button("Start Processing Now", type="primary"):
            start_placeholder.empty()
            try:
                success = preprocess_videos(
                    st.session_state.video_folder, progress_bar, status_text
                )
                if success:
                    st.session_state.processing_done = True
                    st.rerun()
            except Exception as e:
                status_text.error(f"Fehler bei der Verarbeitung: {e}")

# Triage Mode
elif st.session_state.app_mode == "triage":
    st.header("2. Bilder Zuweisen")

    images = sorted((Path(OUTPUT_FOLDER) / "out").glob("*.jpg"))

    if not images:
        st.warning(
            f"Keine Bilder im Ordner {Path(OUTPUT_FOLDER) / 'out'} gefunden. Bitte zuerst Videos verarbeiten."
        )
        if st.button("Zurück zum Setup"):
            st.session_state.app_mode = "setup"
            st.rerun()
        st.stop()

    if st.session_state.index >= len(images):
        save_csv()
        st.success(
            f"Fertig! {len(st.session_state.results)} Einträge gespeichert in `{CSV_PATH}`."
        )
        st.dataframe(
            [
                {"Zeitstempel": r[0], "Aktion": r[1], "Name": r[2]}
                for r in st.session_state.results
            ],
            use_container_width=True,
        )
        if st.button("Neustart"):
            st.session_state.index = 0
            st.session_state.results = []
            st.session_state.app_mode = "setup"
            st.rerun()
    else:
        current_image = images[st.session_state.index]
        timestamp = extract_timestamp(current_image.name)

        # UI Header & Progress
        st.progress(st.session_state.index / len(images))
        st.write(
            f"Bild **{st.session_state.index + 1}** von **{len(images)}** — Zeitstempel: `{timestamp}`"
        )

        # 2-Column Layout
        col_img, col_controls = st.columns([2, 1])

        with col_img:
            st.image(str(current_image), use_container_width=True)

        with col_controls:
            # The current state of selected names
            # Using accept_new_options=True allows users to type custom names
            selected_names = st.multiselect(
                "Namen auswählen (oder neu tippen)",
                options=st.session_state.names,
                placeholder="Namen wählen...",
                accept_new_options=True,
                key=f"names_select_{st.session_state.index}",  # Unique key per image
            )

            # Check if any dynamically typed names need to be added to the persistent list
            new_names = [n for n in selected_names if n not in st.session_state.names]
            if new_names:
                st.session_state.names.extend(new_names)
                # Sort alphabetically for convenience
                st.session_state.names = sorted(list(set(st.session_state.names)))
                save_names(st.session_state.names)

            st.write("---")
            st.subheader("Aktion")

            # Helper to record results and move to next image
            def record_action_and_next(action_type):
                if action_type != "Fehler":
                    for name in selected_names:
                        st.session_state.results.append((timestamp, action_type, name))
                else:
                    # For Fehler, append without a name
                    st.session_state.results.append((timestamp, action_type, ""))

                st.session_state.index += 1

                # Cleanup the multiselect key for this index to avoid state pollution
                for key in list(st.session_state.keys()):
                    if isinstance(key, str) and key.startswith("names_select_"):
                        del st.session_state[key]

                st.rerun()

            # Action Buttons Layout
            action_col1, action_col2 = st.columns(2)

            with action_col1:
                # Disable buttons if no name is selected and the action is not Fehler
                if st.button(
                    "Kommt ⬇️",
                    type="primary",
                    use_container_width=True,
                    disabled=len(selected_names) == 0,
                ):
                    record_action_and_next("Kommt")

            with action_col2:
                if st.button(
                    "Geht ⬆️",
                    type="primary",
                    use_container_width=True,
                    disabled=len(selected_names) == 0,
                ):
                    record_action_and_next("Geht")

            if st.button("Fehler ❌", use_container_width=True):
                record_action_and_next("Fehler")

            st.write("---")

            # Undo button
            if st.session_state.index > 0:
                if st.button("Zurück (Rückgängig) ↩️", use_container_width=True):
                    # We need to go back 1 index and remove all results matching the previous timestamp
                    st.session_state.index -= 1
                    prev_image = images[st.session_state.index]
                    prev_timestamp = extract_timestamp(prev_image.name)

                    st.session_state.results = [
                        r for r in st.session_state.results if r[0] != prev_timestamp
                    ]

                    # Also cleanup current multiselect key to force a fresh render on the previous image
                    for key in list(st.session_state.keys()):
                        if isinstance(key, str) and key.startswith("names_select_"):
                            del st.session_state[key]

                    st.rerun()
