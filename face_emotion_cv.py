#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   Advanced Face Recognition & Emotion Detection System v2   ║
║   NOW WITH: Email alerts + Flask dashboard integration       ║
╚══════════════════════════════════════════════════════════════╝

NEW FEATURES:
  ✅ Email alert when unknown face detected
  ✅ Saves snapshot of unknown face and attaches to email
  ✅ Cooldown so you don't get spammed with emails
  ✅ Writes live data to JSON file for Flask dashboard

SETUP:
  1. Fill in your email details in the CONFIG section below
  2. For Gmail: use an App Password (not your real password)
     Go to: myaccount.google.com → Security → 2-Step Verification → App Passwords
     Create one called "FaceCV" and paste it below

INSTALL:
  pip install opencv-python deepface face_recognition numpy pandas flask

USAGE:
  Run this script:        python face_emotion_cv.py
  Run dashboard separately: python dashboard.py
"""

import cv2
import numpy as np
import os
import csv
import time
import sys
import json
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from pathlib import Path

# ── Dependency check ─────────────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg in ["deepface", "face_recognition", "pandas"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ Missing: pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import face_recognition
from deepface import DeepFace
import pandas as pd

# ╔══════════════════════════════════════════════════════════════╗
# ║                    EMAIL CONFIG                              ║
# ║  Fill these in to enable email alerts                        ║
# ╚══════════════════════════════════════════════════════════════╝
EMAIL_ENABLED       = False          # Set to True once you fill in details below
EMAIL_SENDER        = "your_email@gmail.com"        # Your Gmail address
EMAIL_PASSWORD      = "your_app_password_here"      # Gmail App Password (16 chars)
EMAIL_RECEIVER      = "alert_receiver@gmail.com"    # Where to send alerts
EMAIL_COOLDOWN_SECS = 60             # Minimum seconds between alert emails

# ── File paths ────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR   = Path("known_faces")
ATTENDANCE_FILE   = Path("attendance_log.csv")
SCREENSHOTS_DIR   = Path("screenshots")
UNKNOWN_FACES_DIR = Path("unknown_faces")
LIVE_DATA_FILE    = Path("live_data.json")   # Read by Flask dashboard

# ── Constants ─────────────────────────────────────────────────────────────────
ANALYSIS_EVERY_N  = 5
FACE_MATCH_TOL    = 0.5

EMOTION_COLORS = {
    "happy":    (0,   220, 100),
    "sad":      (200,  80,  40),
    "angry":    (30,   30, 220),
    "fear":     (160,  60, 200),
    "surprise": (0,   200, 255),
    "disgust":  (40,  160,  40),
    "neutral":  (180, 180, 180),
}

WMO_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]


# ── Email Alert System ────────────────────────────────────────────────────────
class EmailAlerter:
    def __init__(self):
        self.last_sent_time = 0
        UNKNOWN_FACES_DIR.mkdir(exist_ok=True)

    def send_alert(self, frame: np.ndarray):
        """Send email alert with snapshot. Runs in background thread."""
        if not EMAIL_ENABLED:
            return
        now = time.time()
        if now - self.last_sent_time < EMAIL_COOLDOWN_SECS:
            return  # Cooldown active, skip
        self.last_sent_time = now

        # Save snapshot
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = UNKNOWN_FACES_DIR / f"unknown_{ts}.jpg"
        cv2.imwrite(str(img_path), frame)

        # Send in background so it doesn't freeze the video
        thread = threading.Thread(target=self._send, args=(str(img_path), ts), daemon=True)
        thread.start()

    def _send(self, img_path: str, timestamp: str):
        try:
            msg = MIMEMultipart()
            msg["Subject"] = f"⚠️ Unknown Face Detected — {timestamp}"
            msg["From"]    = EMAIL_SENDER
            msg["To"]      = EMAIL_RECEIVER

            body = MIMEText(f"""
Hello,

An unknown face was detected by your Face Recognition System.

Time: {timestamp}
A snapshot has been attached to this email.

— FaceCV Security System
""")
            msg.attach(body)

            # Attach snapshot
            with open(img_path, "rb") as f:
                img_attachment = MIMEImage(f.read(), name=os.path.basename(img_path))
                msg.attach(img_attachment)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

            print(f"  📧 Alert email sent to {EMAIL_RECEIVER}")
        except Exception as e:
            print(f"  ⚠️  Email failed: {e}")


# ── Live Data Writer (for Flask dashboard) ────────────────────────────────────
class LiveDataWriter:
    def __init__(self):
        self.session_start   = datetime.now().isoformat()
        self.emotion_counts  = {e: 0 for e in WMO_EMOTIONS}
        self.recognized_log  = []   # [{name, time, emotion, age, gender}]
        self.unknown_count   = 0
        self.frame_count     = 0
        self._write()

    def update(self, faces: list[dict], fps: float):
        self.frame_count += 1
        for f in faces:
            emo = f.get("emotion", "neutral")
            if emo in self.emotion_counts:
                self.emotion_counts[emo] += 1
            if f.get("name", "Unknown") != "Unknown":
                self.recognized_log.append({
                    "name":    f["name"],
                    "time":    datetime.now().strftime("%H:%M:%S"),
                    "emotion": emo,
                    "age":     f.get("age", "?"),
                    "gender":  f.get("gender", "?"),
                })
                # Keep last 50 entries
                self.recognized_log = self.recognized_log[-50:]
            else:
                self.unknown_count += 1
        self._write(fps=fps, active_faces=len(faces))

    def _write(self, fps: float = 0.0, active_faces: int = 0):
        data = {
            "session_start":  self.session_start,
            "last_updated":   datetime.now().isoformat(),
            "fps":            round(fps, 1),
            "active_faces":   active_faces,
            "unknown_count":  self.unknown_count,
            "emotion_counts": self.emotion_counts,
            "recognized_log": self.recognized_log,
        }
        try:
            with open(LIVE_DATA_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


# ── Face Database ─────────────────────────────────────────────────────────────
class FaceDatabase:
    def __init__(self):
        KNOWN_FACES_DIR.mkdir(exist_ok=True)
        self.encodings: list = []
        self.names:     list = []
        self.load()

    def load(self):
        self.encodings, self.names = [], []
        for img_path in KNOWN_FACES_DIR.glob("*.jpg"):
            img  = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img)
            if encs:
                self.encodings.append(encs[0])
                self.names.append(img_path.stem.split("_")[0])
        print(f"  👤 Loaded {len(self.names)} known face(s): {self.names}")

    def register(self, frame: np.ndarray, name: str) -> bool:
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if not encs:
            print("  ❌ No face detected.")
            return False
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = KNOWN_FACES_DIR / f"{name}_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        self.encodings.append(encs[0])
        self.names.append(name)
        print(f"  ✅ Registered '{name}'")
        return True

    def identify(self, encoding) -> tuple[str, float]:
        if not self.encodings:
            return "Unknown", 0.0
        distances = face_recognition.face_distance(self.encodings, encoding)
        best_idx  = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        conf      = max(0.0, 1.0 - best_dist)
        if best_dist <= FACE_MATCH_TOL:
            return self.names[best_idx], conf
        return "Unknown", conf


# ── Attendance Logger ─────────────────────────────────────────────────────────
class AttendanceLogger:
    def __init__(self):
        self.logged_today: set = set()
        if not ATTENDANCE_FILE.exists():
            with open(ATTENDANCE_FILE, "w", newline="") as f:
                csv.writer(f).writerow(["Name","Date","Time","Emotion","Age","Gender"])

    def log(self, name, emotion, age, gender):
        if name == "Unknown" or name in self.logged_today:
            return
        now = datetime.now()
        self.logged_today.add(name)
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            csv.writer(f).writerow([name, now.strftime("%Y-%m-%d"),
                                    now.strftime("%H:%M:%S"), emotion, age, gender])
        print(f"  📋 Logged: {name}")


# ── HUD Helpers ───────────────────────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, radius=10):
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    for cx, cy, s, e in [(x1+radius,y1+radius,180,270),(x2-radius,y1+radius,270,360),
                          (x1+radius,y2-radius,90,180),(x2-radius,y2-radius,0,90)]:
        cv2.ellipse(img,(cx,cy),(radius,radius),0,s,e,color,thickness)

def draw_filled_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_emotion_bar(img, x, y, emotion, score, bar_width=120):
    color  = EMOTION_COLORS.get(emotion, (180,180,180))
    filled = int(bar_width * score)
    cv2.rectangle(img, (x,y), (x+bar_width, y+10), (50,50,50), -1)
    cv2.rectangle(img, (x,y), (x+filled,    y+10), color,      -1)
    cv2.putText(img, f"{emotion[:7]:7s} {score*100:4.0f}%",
                (x+bar_width+5, y+9), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220,220,220), 1)

def text(img, t, pos, scale=0.55, color=(255,255,255), thickness=1):
    x, y = pos
    cv2.putText(img, t, (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, t, (x,  y),   cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thickness,   cv2.LINE_AA)

def wind_dir(deg):
    return ["N","NE","E","SE","S","SW","W","NW"][round(deg/45)%8]


# ── Main App ──────────────────────────────────────────────────────────────────
class FaceEmotionApp:
    def __init__(self):
        self.db            = FaceDatabase()
        self.logger        = AttendanceLogger()
        self.alerter       = EmailAlerter()
        self.live_data     = LiveDataWriter()
        SCREENSHOTS_DIR.mkdir(exist_ok=True)

        self.show_emotion    = True
        self.show_age_gender = True
        self.logging_active  = True

        self.frame_count   = 0
        self.fps           = 0.0
        self.fps_timer     = time.time()
        self.fps_frames    = 0

        self.analysis_cache: dict = {}
        self.prev_locations: list = []

        status = "ENABLED ✅" if EMAIL_ENABLED else "DISABLED (set EMAIL_ENABLED=True in config)"
        print(f"  📧 Email alerts: {status}")
        print(f"  📊 Dashboard data: {LIVE_DATA_FILE}")

    def _face_id(self, loc) -> int:
        top, right, bottom, left = loc
        cx, cy = (left+right)//2, (top+bottom)//2
        for i, (_, fy, fx) in enumerate(self.prev_locations):
            if abs(cx-fx) < 80 and abs(cy-fy) < 80:
                return i
        return len(self.prev_locations)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  ❌ Cannot open webcam.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("  ✅ Camera ready!")
        print("  [R] Register  [S] Screenshot  [A] Attendance  [E] Emotion  [G] Age/Gender  [Q] Quit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count  += 1
            self.fps_frames   += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Face detection ───────────────────────────────────────────
            small     = cv2.resize(rgb, (0,0), fx=0.5, fy=0.5)
            face_locs = face_recognition.face_locations(small, model="hog")
            face_locs = [(t*2,r*2,b*2,l*2) for t,r,b,l in face_locs]
            face_encs = face_recognition.face_encodings(rgb, face_locs)

            new_prev   = []
            frame_data = []   # Collect per-face info for live_data

            for loc, enc in zip(face_locs, face_encs):
                top, right, bottom, left = loc
                face_id = self._face_id(loc)
                cx, cy  = (left+right)//2, (top+bottom)//2
                new_prev.append((loc, cy, cx))

                # Identity
                name, conf = self.db.identify(enc)

                # Email alert for unknowns
                if name == "Unknown":
                    self.alerter.send_alert(frame)

                # DeepFace every N frames
                if self.frame_count % ANALYSIS_EVERY_N == 0 or face_id not in self.analysis_cache:
                    try:
                        crop = frame[max(0,top-20):bottom+20, max(0,left-20):right+20]
                        if crop.size > 0:
                            r = DeepFace.analyze(crop, actions=["emotion","age","gender"],
                                                 enforce_detection=False, silent=True)[0]
                            self.analysis_cache[face_id] = {
                                "emotion":        r["dominant_emotion"],
                                "emotion_scores": r["emotion"],
                                "age":            str(r["age"]),
                                "gender":         r["dominant_gender"],
                            }
                    except Exception:
                        if face_id not in self.analysis_cache:
                            self.analysis_cache[face_id] = {
                                "emotion":"neutral","emotion_scores":{},"age":"?","gender":"?"
                            }

                info    = self.analysis_cache.get(face_id, {})
                emotion = info.get("emotion", "neutral")
                scores  = info.get("emotion_scores", {})
                age     = info.get("age", "?")
                gender  = info.get("gender", "?")

                frame_data.append({"name": name, "emotion": emotion, "age": age, "gender": gender})

                if self.logging_active:
                    self.logger.log(name, emotion, age, gender)

                # ── Draw ─────────────────────────────────────────────────
                color = EMOTION_COLORS.get(emotion, (180,180,180))
                draw_rounded_rect(frame, left, top, right, bottom, color, 2)

                # Corner accents
                for ax, ay, dx, dy in [(left,top,1,1),(right,top,-1,1),(left,bottom,1,-1),(right,bottom,-1,-1)]:
                    cv2.line(frame, (ax,ay), (ax+dx*20, ay),    color, 3)
                    cv2.line(frame, (ax,ay), (ax, ay+dy*20),    color, 3)

                # Name banner
                by = max(0, top-32)
                label = f"{name}  {conf*100:.0f}%" if name != "Unknown" else "⚠ Unknown"
                (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                banner_color = color if name != "Unknown" else (30,30,200)
                draw_filled_rect(frame, left, by, left+tw+12, by+th+10, banner_color, 0.75)
                text(frame, label, (left+6, by+th+2), scale=0.6, color=(255,255,255))

                # Emotion bars
                if self.show_emotion and scores:
                    for ei, (emo, sc) in enumerate(sorted(scores.items(), key=lambda x:-x[1])[:5]):
                        draw_emotion_bar(frame, right+10, top+ei*18, emo, sc/100)

                # Age/gender
                if self.show_age_gender:
                    gy = top + (5*18 if self.show_emotion else 0) + 8
                    text(frame, f"Age: {age}",    (right+10, gy),    scale=0.5)
                    text(frame, f"Sex: {gender}", (right+10, gy+18), scale=0.5)

            self.prev_locations = new_prev

            # ── FPS ───────────────────────────────────────────────────────
            now = time.time()
            if now - self.fps_timer >= 1.0:
                self.fps       = self.fps_frames / (now - self.fps_timer + 1e-9)
                self.fps_timer  = now
                self.fps_frames = 0

            # Update live data for dashboard
            self.live_data.update(frame_data, self.fps)

            # ── HUD ───────────────────────────────────────────────────────
            hud = [f"FPS: {self.fps:.1f}", f"Faces: {len(face_locs)}",
                   f"Known: {len(self.db.names)}", f"Log: {'ON' if self.logging_active else 'OFF'}",
                   f"Email: {'ON' if EMAIL_ENABLED else 'OFF'}"]
            draw_filled_rect(frame, 8, 8, 165, 8+len(hud)*22, (20,20,20), 0.65)
            for i, line in enumerate(hud):
                color = (0,230,120) if "ON" in line else (180,180,180)
                text(frame, line, (14, 26+i*22), scale=0.52, color=color)

            h, w = frame.shape[:2]
            hint = "[R] Register  [S] Screenshot  [A] Attendance  [E] Emotion  [G] Age/Gender  [Q] Quit"
            draw_filled_rect(frame, 0, h-28, w, h, (10,10,10), 0.7)
            text(frame, hint, (10, h-10), scale=0.42, color=(160,160,160))

            cv2.imshow("Face Recognition & Emotion Detection v2", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                print("\n  📸 REGISTER NEW FACE")
                name_in = input("  Enter name: ").strip()
                if name_in:
                    ret2, snap = cap.read()
                    if ret2:
                        self.db.register(snap, name_in)
            elif key == ord("s"):
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = SCREENSHOTS_DIR / f"capture_{ts}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"  📷 Screenshot: {path}")
            elif key == ord("a"):
                self.logging_active = not self.logging_active
                print(f"  📋 Attendance: {'ON' if self.logging_active else 'OFF'}")
            elif key == ord("e"):
                self.show_emotion = not self.show_emotion
            elif key == ord("g"):
                self.show_age_gender = not self.show_age_gender

        cap.release()
        cv2.destroyAllWindows()
        print("\n  👋 Session ended.")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Face Recognition & Emotion Detection v2                   ║
║   Email Alerts + Dashboard Integration                       ║
╚══════════════════════════════════════════════════════════════╝
""")
    FaceEmotionApp().run()
