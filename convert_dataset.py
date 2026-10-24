"""
convert_dataset.py  (v2 — improved accuracy)
=============================================
Converts the ISL_CSLRT_Corpus Frames_Word_Level dataset into the
preset-references.json format that the Sixth-Sense website uses.

HOW TO RUN:
  pip install mediapipe opencv-python
  python convert_dataset.py

WHY v2 IS MORE ACCURATE:
  Each dataset image is a static photo of a sign being performed.
  The website does recognition by comparing your live webcam frames
  (captured over ~900ms) against stored "takes".

  v1 grouped random images together into takes — not ideal because
  the images are unrelated snapshots, not sequential video frames.

  v2 treats EACH IMAGE as its own take by repeating the landmark
  frame 20 times. This perfectly simulates someone HOLDING a sign
  still for 900ms, which is exactly what the DTW matcher expects
  when comparing against a held static pose.
"""

import os
import json
import math
import sys
import cv2
import mediapipe as mp

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\gupta\Desktop\sign-lan\isl-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Word_Level"
OUTPUT_FILE = r"C:\Users\gupta\Desktop\mini3\Sixth-Sense\data\preset-references.json"

# How many frames to repeat per image to simulate a "held" sign.
REPEAT_FRAMES = 20

# Max takes per word (each image = one take; cap to keep file small)
MAX_TAKES_PER_WORD = 3

# Minimum image file size in bytes to skip tiny thumbnail images
MIN_IMAGE_SIZE_BYTES = 15000

# ──────────────────────────────────────────────────────────────
# WORD ALIASES — dataset folders have combined names like "HELLO_HI"
# but users type "HELLO". This maps folder names to extra word entries.
# ──────────────────────────────────────────────────────────────
WORD_ALIASES = {
    "HELLO_HI":       ["HELLO", "HI"],
    "I_ME_MINE_MY":   ["I", "ME", "MINE", "MY"],
    "LIKE_LOVE":      ["LIKE", "LOVE"],
    "COLLEGE_SCHOOL": ["COLLEGE", "SCHOOL"],
    "OLD_AGE":        ["OLD", "AGE"],
    "DON'T CARE":     ["DONT CARE"],
    "SOME ONE":       ["SOMEONE"],
    "SOME HOW":       ["SOMEHOW"],
    "SOME MORE":      ["MORE"],
    "SO MUCH":        ["MUCH"],
    "THIS ONE":       ["THIS"],
    "TAKE CARE":      ["CARE"],
    "TAKE TIME":      ["TIME"],
    "TURN ON":        ["ON"],
    "ON THE WAY":     ["WAY"],
    "MEAN IT":        ["MEAN"],
    "A LOT":          ["LOT"],
}

# ──────────────────────────────────────────────────────────────
# These constants MUST match recognition.js exactly
# ──────────────────────────────────────────────────────────────
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
FACE_IDX = [
    1,
    61, 185, 40, 0, 270, 409, 291, 405, 17, 181,
    70, 105, 107, 300, 334, 336,
    33, 160, 158, 133, 153, 144,
    263, 387, 385, 362, 380, 373,
    103, 10, 332, 454, 397, 172, 234
]


def normalize_hand(hand_landmarks):
    """Match normalizeHand() in recognition.js — normalizes against wrist and palm size."""
    if hand_landmarks is None or len(hand_landmarks) < 10:
        return None
    wrist = hand_landmarks[0]
    mid_mcp = hand_landmarks[9]
    scale = max(0.02, math.hypot(wrist.x - mid_mcp.x, wrist.y - mid_mcp.y))
    vec = []
    for p in hand_landmarks:
        vec.append(-(p.x - wrist.x) / scale)
        vec.append((p.y - wrist.y) / scale)
        vec.append((p.z or 0.0) / scale)
    return vec


def normalize_set(points, idx_list, origin, scale):
    """Match normalizeSet() in recognition.js."""
    if points is None or scale < 1e-4:
        return None
    vec = []
    for i in idx_list:
        if i >= len(points):
            return None
        p = points[i]
        vec.append(-(p.x - origin['x']) / scale)
        vec.append((p.y - origin['y']) / scale)
        vec.append((p.z or 0.0) / scale)
    return vec


def extract_frame(results):
    """Extract one frame dict {pose, left, right, face} from MediaPipe results."""
    pose_lms = results.pose_landmarks
    left_lms = results.left_hand_landmarks
    right_lms = results.right_hand_landmarks
    face_lms = results.face_landmarks

    pose_vec = None
    if pose_lms and len(pose_lms.landmark) > 24:
        lm = pose_lms.landmark
        ls, rs = lm[11], lm[12]
        ox = (ls.x + rs.x) / 2
        oy = (ls.y + rs.y) / 2
        scale = max(0.05, math.hypot(ls.x - rs.x, ls.y - rs.y))
        pose_vec = normalize_set(lm, POSE_IDX, {'x': ox, 'y': oy}, scale)

    left_vec  = normalize_hand(left_lms.landmark  if left_lms  else None)
    right_vec = normalize_hand(right_lms.landmark if right_lms else None)

    face_vec = None
    if face_lms:
        lm = face_lms.landmark
        if len(lm) > 362 and lm[133] and lm[362] and lm[1]:
            origin = {'x': lm[1].x, 'y': lm[1].y}
            scale = max(0.03, math.hypot(lm[133].x - lm[362].x, lm[133].y - lm[362].y))
            face_vec = normalize_set(lm, FACE_IDX, origin, scale)

    return {'pose': pose_vec, 'left': left_vec, 'right': right_vec, 'face': face_vec}


def process_word_folder(word, folder_path, holistic):
    """
    KEY CHANGE from v1:
      Each usable image → one take of REPEAT_FRAMES identical frames.
      This mimics a person HOLDING the sign still in front of the webcam,
      which is exactly what DTW matching expects to see.
    """
    # Sort files; skip tiny thumbnails (they're too small to detect hands in)
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and os.path.getsize(os.path.join(folder_path, f)) >= MIN_IMAGE_SIZE_BYTES
    ])

    if not image_files:
        return []

    takes = []
    for fname in image_files:
        img = cv2.imread(os.path.join(folder_path, fname))
        if img is None:
            continue

        results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        frame = extract_frame(results)

        # Skip if no hands detected in this image
        if not frame['right'] and not frame['left']:
            continue

        meta = {
            'right': frame['right'] is not None,
            'left':  frame['left']  is not None,
            'face':  frame['face']  is not None,
        }

        # Repeat this single frame REPEAT_FRAMES times →
        # simulates holding the sign still during a live recording
        frames = [frame] * REPEAT_FRAMES

        takes.append({'meta': meta, 'frames': frames})

        if len(takes) >= MAX_TAKES_PER_WORD:
            break

    return takes


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: Dataset folder not found:\n  {DATASET_DIR}")
        sys.exit(1)

    word_folders = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    print(f"Found {len(word_folders)} word folders in dataset.")
    print(f"Strategy: each image → {REPEAT_FRAMES}-frame 'held pose' take (max {MAX_TAKES_PER_WORD}/word)\n")

    references = {}

    with mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.4,
    ) as holistic:
        for i, word_folder in enumerate(word_folders, 1):
            folder_path = os.path.join(DATASET_DIR, word_folder)
            word = word_folder.upper()

            print(f"[{i:3d}/{len(word_folders)}] {word} ...", end=' ', flush=True)
            takes = process_word_folder(word, folder_path, holistic)

            if takes:
                references[word] = takes
                parts = []
                if takes[0]['meta']['right']: parts.append('R')
                if takes[0]['meta']['left']:  parts.append('L')
                if takes[0]['meta']['face']:  parts.append('F')
                print(f"✓ {len(takes)} take(s) [{'+'.join(parts)}]")
            else:
                print("✗ skipped (no hands detected in usable images)")

    if not references:
        print("\nNo words processed. Check the dataset path.")
        sys.exit(1)

    # Expand aliases: HELLO_HI → also create HELLO and HI entries
    alias_count = 0
    for original, aliases in WORD_ALIASES.items():
        if original in references:
            for alias in aliases:
                if alias not in references:
                    references[alias] = references[original]
                    alias_count += 1
    if alias_count:
        print(f"\n+ Added {alias_count} word aliases (HELLO, HI, LIKE, etc.)")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(references, f, separators=(',', ':'))

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n✅ Done! {len(references)} words → {OUTPUT_FILE}  ({size_mb:.1f} MB)")
    print("Reload the website — all signs load automatically!")


if __name__ == '__main__':
    main()
