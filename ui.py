import csv
import streamlit as st
from pathlib import Path

OUTPUT_FOLDER = "camsp"
CSV_PATH = Path(OUTPUT_FOLDER) / "ergebnisse.csv"
NAMES_PATH = Path("names.txt")

# Bilder laden und nach Name sortieren
images = sorted((Path(OUTPUT_FOLDER) / "out").glob("*.jpg"))

# Namen laden
names = [n.strip() for n in NAMES_PATH.read_text().splitlines() if n.strip()]

# Session-State initialisieren
if "index" not in st.session_state:
    st.session_state.index = 0
if "results" not in st.session_state:
    st.session_state.results = []
if "action" not in st.session_state:
    st.session_state.action = None


def extract_timestamp(filename: str) -> str:
    """Extracts the ISO timestamp after the last underscore in the filename."""
    stem = Path(filename).stem  # ohne .jpg
    return stem.rsplit("_", 1)[-1]


def save_csv():
    """Writes results list to CSV."""
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "action", "name"])
        writer.writerows(st.session_state.results)


st.title("MonkeyCam - Zuweisungstool")

if st.session_state.index >= len(images):
    # Alle Bilder bearbeitet
    save_csv()
    st.success(f"Fertig! {len(st.session_state.results)} Einträge gespeichert in `{CSV_PATH}`.")
    st.dataframe(
        [{"Zeitstempel": r[0], "Aktion": r[1], "Name": r[2]} for r in st.session_state.results],
        width="stretch",
    )
    if st.button("Neustart"):
        st.session_state.index = 0
        st.session_state.results = []
        st.session_state.action = None
        st.rerun()
else:
    current_image = images[st.session_state.index]
    timestamp = extract_timestamp(current_image.name)

    st.write(f"Bild **{st.session_state.index + 1}** von **{len(images)}** — Zeitstempel: `{timestamp}`")
    st.image(str(current_image), width="stretch")

    options = ["Kommt", "Geht", "Fehler"]
    action = st.segmented_control("Aktion wählen", options, selection_mode="single")

    # Name selection — only needed for Kommt/Geht
    name = None
    if action in ("Kommt", "Geht"):
        name = st.selectbox("Name", options=names, index=None, placeholder="Name wählen...")

    # Weiter enabled when action is set AND (Fehler OR name chosen)
    can_continue = action is not None and (action == "Fehler" or name is not None)

    if st.button("Weiter", type="primary", disabled=not can_continue):
        if action != "Fehler":
            st.session_state.results.append((timestamp, action, name))
        st.session_state.index += 1
        st.session_state.action = None
        st.rerun()
    
