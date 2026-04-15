"""
Virtual Paint / Gesture Drawing Application
============================================
Uses OpenCV + MediaPipe for real-time hand-gesture-based drawing.

Gesture Reference
-----------------
  Drawing Mode   : Index finger only raised  → draw with fingertip
  Selection Mode : Index + Middle fingers up → hover / pick colour or tool
  Save & Clear   : Fist (all fingers closed) → save current drawing, clear canvas
  Quit           : Press 'q'                 → exit at any time

Run:
    pip install opencv-python mediapipe numpy
    python virtual_paint.py
"""

import sys
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import mediapipe as mp
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
FRAME_W, FRAME_H   = 1280, 720
THUMB_W, THUMB_H   = 240, 180          # gallery thumbnail size
MAX_ITEMS          = 3                 # total drawings to capture
HEADER_H           = 110              # top toolbar height
SMOOTHING          = 0.55              # EWMA alpha: 0=max-smooth/lag, 1=raw/sharp
GESTURE_HISTORY    = 7                 # frames for gesture majority-vote stabiliser
FIST_THRESH        = 28               # hold fist for this many frames to save
SAVE_COOLDOWN      = 40               # frames to ignore gestures after a save

# Colour palette  (BGR)
COLOURS = {
    "Red"    : (0,   0,   220),
    "Green"  : (0,   200, 0  ),
    "Blue"   : (220, 0,   0  ),
    "Yellow" : (0,   215, 255),
    "White"  : (255, 255, 255),
    "Eraser" : (0,   0,   0  ),
}
BRUSH_SIZES   = [5, 10, 20, 35]
COLOUR_NAMES  = list(COLOURS.keys())

# UI layout for the top toolbar
COLOUR_RECTS = []          # filled in VirtualPaint.__init__
BRUSH_RECTS  = []

# ──────────────────────────────────────────────────────────────────────────────
# Permission Dialog
# ──────────────────────────────────────────────────────────────────────────────

def ask_camera_permission() -> bool:
    """
    Show a GUI dialog asking the user for webcam access.
    Returns True if the user agrees, False otherwise.
    Falls back to a console prompt if tkinter is unavailable.
    """
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        result = messagebox.askyesno(
            title="Camera Permission",
            message=(
                "Virtual Paint requires access to your webcam.\n\n"
                "Do you allow this application to use your camera?"
            ),
        )
        root.destroy()
        return result
    except Exception:
        ans = input("\n[Permission] Allow webcam access? (yes/no): ").strip().lower()
        return ans in ("yes", "y")


# ──────────────────────────────────────────────────────────────────────────────
# Hand Detector
# ──────────────────────────────────────────────────────────────────────────────

class HandDetector:
    """Thin wrapper around MediaPipe Hands."""

    FINGER_TIPS = [4, 8, 12, 16, 20]   # landmark IDs for each fingertip
    FINGER_PIPS = [3, 6, 10, 14, 18]   # one joint below each tip

    def __init__(self, max_hands: int = 1, detection_conf: float = 0.6,
                 tracking_conf: float = 0.6):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,          # more accurate landmark model
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._draw = mp.solutions.drawing_utils
        self.results = None
        self.landmarks: list = []        # pixel coords of all 21 landmarks

    def process(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Run hand detection and return the annotated frame."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self._hands.process(rgb)
        rgb.flags.writeable = True
        self.landmarks = []

        if self.results.multi_hand_landmarks:
            hand_lms = self.results.multi_hand_landmarks[0]
            h, w = bgr_frame.shape[:2]
            self.landmarks = [
                (int(lm.x * w), int(lm.y * h))
                for lm in hand_lms.landmark
            ]
            self._draw.draw_landmarks(
                bgr_frame, hand_lms,
                self._mp_hands.HAND_CONNECTIONS,
                self._draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=4),
                self._draw.DrawingSpec(color=(255, 200, 0), thickness=2),
            )
        return bgr_frame

    def fingers_up(self) -> list[int]:
        """
        Return a list of 5 booleans (0/1) indicating which fingers are extended.
        Order: [Thumb, Index, Middle, Ring, Pinky]

        Uses MCP (knuckle) joints as a reference base so detection stays
        accurate even when the hand is near the bottom or edges of the frame
        (where wrist landmarks can be extrapolated incorrectly by MediaPipe).
        """
        if len(self.landmarks) < 21:
            return [0, 0, 0, 0, 0]

        up = []

        # ── Thumb ─────────────────────────────────────────────────────────
        # Measure horizontal distance from thumb MCP (2) instead of wrist;
        # this is robust to partial-frame / bottom-edge detections.
        thumb_mcp = self.landmarks[2]
        thumb_ip  = self.landmarks[3]
        thumb_tip = self.landmarks[4]
        d_tip = abs(thumb_tip[0] - thumb_mcp[0])
        d_ip  = abs(thumb_ip[0]  - thumb_mcp[0])
        up.append(1 if d_tip > d_ip else 0)

        # ── Other 4 fingers ────────────────────────────────────────────────
        # A finger is "up" when its tip is above both its PIP joint AND its
        # MCP (knuckle) joint.  Requiring two conditions prevents false "up"
        # readings caused by extrapolated landmarks near frame edges.
        TIPS = [8, 12, 16, 20]
        PIPS = [6, 10, 14, 18]
        MCPS = [5,  9, 13, 17]
        for tip_id, pip_id, mcp_id in zip(TIPS, PIPS, MCPS):
            tip_y = self.landmarks[tip_id][1]
            pip_y = self.landmarks[pip_id][1]
            mcp_y = self.landmarks[mcp_id][1]
            # Both conditions must hold; second prevents marginal readings.
            extended = (tip_y < pip_y) and (tip_y < mcp_y)
            up.append(1 if extended else 0)
        return up

    @property
    def index_tip(self):
        """Pixel coordinate of index finger tip (landmark 8)."""
        if len(self.landmarks) >= 9:
            return self.landmarks[8]
        return None

    def palm_hull(self) -> np.ndarray | None:
        """
        Return the convex hull of the palm region as a contour array
        suitable for cv2.fillConvexPoly / cv2.polylines.

        Uses the wrist + all four finger MCP (knuckle) joints + thumb CMC
        to outline the palm without including the long finger bones, giving
        a tight region that matches the physical palm surface.
        """
        if len(self.landmarks) < 21:
            return None
        # Landmark IDs that trace the palm boundary:
        #   0  = wrist
        #   1  = thumb CMC (base of thumb)
        #   5  = index MCP
        #   9  = middle MCP
        #   13 = ring MCP
        #   17 = pinky MCP
        PALM_IDS = [0, 1, 2, 5, 9, 13, 17]
        pts = np.array([self.landmarks[i] for i in PALM_IDS], dtype=np.int32)
        return cv2.convexHull(pts)


# ──────────────────────────────────────────────────────────────────────────────
# Toolbar / UI renderer
# ──────────────────────────────────────────────────────────────────────────────

class Toolbar:
    """Renders the top colour/brush toolbar onto a BGR image."""

    PAD = 8

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self._build_rects()

    def _build_rects(self):
        """Pre-compute bounding boxes for each colour & brush button."""
        n_colours = len(COLOUR_NAMES)
        colour_btn_w = 120
        total_colour_w = n_colours * (colour_btn_w + self.PAD) + self.PAD
        start_x = self.PAD

        self.colour_rects = []
        for i, name in enumerate(COLOUR_NAMES):
            x1 = start_x + i * (colour_btn_w + self.PAD)
            self.colour_rects.append({
                "name" : name,
                "bgr"  : COLOURS[name],
                "rect" : (x1, self.PAD, x1 + colour_btn_w, self.h - self.PAD),
            })

        brush_btn_w = 60
        bx = total_colour_w + 20
        self.brush_rects = []
        for i, sz in enumerate(BRUSH_SIZES):
            x1 = bx + i * (brush_btn_w + self.PAD)
            self.brush_rects.append({
                "size" : sz,
                "rect" : (x1, self.PAD, x1 + brush_btn_w, self.h - self.PAD),
            })

        # Save-button rect (right side)
        self.save_rect = (self.w - 180, self.PAD, self.w - self.PAD, self.h - self.PAD)

    def draw(self, frame: np.ndarray, active_colour: str,
             active_brush: int, item_count: int):
        """Paint the toolbar band onto frame (in-place)."""
        # Semi-transparent toolbar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Colour buttons
        for btn in self.colour_rects:
            x1, y1, x2, y2 = btn["rect"]
            is_eraser = btn["name"] == "Eraser"
            is_active  = btn["name"] == active_colour

            if is_eraser:
                # Distinctive eraser button: light grey bg + dashed-border look
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), -1)
                # Diagonal stripes to signal "erase"
                for sx in range(x1, x2, 10):
                    cv2.line(frame, (sx, y1), (min(sx + (y2-y1), x2), y2),
                             (160, 160, 160), 1)
                cv2.putText(frame, "X ERASE", (x1 + 5, y2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
                # Active: red glow border
                border_col = (0, 0, 220) if is_active else (120, 120, 120)
                cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), border_col, 3)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), btn["bgr"], -1)
                if is_active:
                    cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 255), 3)
                label = btn["name"]
                cv2.putText(frame, label, (x1 + 5, y2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Brush-size buttons
        for btn in self.brush_rects:
            x1, y1, x2, y2 = btn["rect"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), -1)
            if btn["size"] == active_brush:
                cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 255), 3)
            cv2.circle(frame, (cx, cy), min(btn["size"] // 2, 20),
                       (200, 200, 200), -1)

        # Save hint
        x1, y1, x2, y2 = self.save_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 180, 50), -1)
        cv2.putText(frame, "✊ SAVE", (x1 + 12, (y1 + y2) // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Item counter
        cv2.putText(frame, f"Items: {item_count}/{MAX_ITEMS}",
                    (self.w // 2 - 70, self.h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2, cv2.LINE_AA)

    def hit_test(self, pt) -> dict | None:
        """
        Given a pixel coordinate pt=(x,y), check if it falls inside a toolbar
        button. Returns a dict with action details or None.
        """
        if pt is None or pt[1] > self.h:
            return None
        for btn in self.colour_rects:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
                return {"type": "colour", "value": btn["name"]}
        for btn in self.brush_rects:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
                return {"type": "brush", "value": btn["size"]}
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Gallery
# ──────────────────────────────────────────────────────────────────────────────

class Gallery:
    """Holds up to MAX_ITEMS thumbnails and renders them."""

    PANEL_W = THUMB_W + 20

    def __init__(self):
        self.items: list[np.ndarray] = []

    def add(self, canvas: np.ndarray):
        thumb = cv2.resize(canvas, (THUMB_W, THUMB_H))
        self.items.append(thumb)

    def draw_sidebar(self, frame: np.ndarray):
        """Draw saved thumbnails on the right edge of frame."""
        x_start = frame.shape[1] - self.PANEL_W
        cv2.rectangle(frame,
                      (x_start, HEADER_H),
                      (frame.shape[1], frame.shape[0]),
                      (20, 20, 20), -1)
        cv2.putText(frame, "Saved", (x_start + 10, HEADER_H + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        for i, thumb in enumerate(self.items):
            y = HEADER_H + 40 + i * (THUMB_H + 15)
            frame[y:y + THUMB_H, x_start + 10:x_start + 10 + THUMB_W] = thumb
            cv2.rectangle(frame,
                          (x_start + 10 - 1, y - 1),
                          (x_start + 10 + THUMB_W + 1, y + THUMB_H + 1),
                          (100, 200, 255), 1)
            cv2.putText(frame, f"#{i+1}", (x_start + 15, y + THUMB_H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

    def build_gallery_frame(self) -> np.ndarray:
        """Render a premium full-screen gallery image (FRAME_W x FRAME_H)."""
        W, H = FRAME_W, FRAME_H
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # ── Deep gradient background: dark navy → deep violet ──────────────
        for row in range(H):
            t = row / H
            b = int(12  + t * 10)
            g = int(8   + t * 6)
            r = int(18  + t * 24)
            frame[row, :] = (b, g, r)

        # ── Subtle horizontal scanline texture ─────────────────────────────
        for row in range(0, H, 4):
            frame[row, :] = np.clip(
                frame[row, :].astype(np.int32) + 6, 0, 255).astype(np.uint8)

        # ── Title bar area ─────────────────────────────────────────────────
        title_h = 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, title_h), (20, 10, 35), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Thin accent line below title
        cv2.line(frame, (0, title_h), (W, title_h), (180, 80, 255), 2)

        # Glowing dot accent left of title
        cv2.circle(frame, (44, title_h // 2), 10, (200, 100, 255), -1)
        cv2.circle(frame, (44, title_h // 2), 16, (160, 60, 220), 2)

        # Title text
        cv2.putText(frame, "YOUR GALLERY",
                    (72, title_h // 2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1,
                    (220, 160, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Gesture  ·  Draw  ·  Create",
                    (74, title_h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (150, 100, 200), 1, cv2.LINE_AA)

        # Press Q hint (top right)
        cv2.putText(frame, "Press  Q  to exit",
                    (W - 230, title_h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (120, 90, 160), 1, cv2.LINE_AA)

        # ── Cards ─────────────────────────────────────────────────────────
        cols       = len(self.items)
        card_pad   = 36
        card_w     = (W - card_pad * (cols + 1)) // cols
        card_h     = H - title_h - card_pad * 2 - 60   # leave room for label
        label_h    = 50
        y_card     = title_h + card_pad

        # Accent colours per card (BGR)
        accent_cols = [
            (255, 120, 80),    # neon coral
            (80,  220, 255),   # cyan
            (160, 100, 255),   # violet
        ]

        for i, thumb in enumerate(self.items):
            x_card = card_pad + i * (card_w + card_pad)
            ac = accent_cols[i % len(accent_cols)]

            # ── Card shadow (offset dark rect) ──────────────────────────
            shadow_offset = 8
            shadow = frame.copy()
            cv2.rectangle(shadow,
                          (x_card + shadow_offset, y_card + shadow_offset),
                          (x_card + card_w + shadow_offset,
                           y_card + card_h + shadow_offset),
                          (5, 5, 10), -1)
            cv2.addWeighted(shadow, 0.6, frame, 0.4, 0, frame)

            # ── Card dark background ─────────────────────────────────────
            card_bg = frame.copy()
            cv2.rectangle(card_bg, (x_card, y_card),
                          (x_card + card_w, y_card + card_h),
                          (22, 15, 38), -1)
            cv2.addWeighted(card_bg, 0.9, frame, 0.1, 0, frame)

            # ── Glowing border (layered) ──────────────────────────────────
            for thickness, alpha, shrink in [(6, 0.25, 0), (2, 0.9, 2)]:
                glow = frame.copy()
                cv2.rectangle(glow,
                              (x_card - shrink, y_card - shrink),
                              (x_card + card_w + shrink,
                               y_card + card_h + shrink),
                              ac, thickness)
                cv2.addWeighted(glow, alpha, frame, 1 - alpha, 0, frame)

            # ── Thumbnail (fit inside card with 12px inset) ───────────────
            inset = 12
            img_x1 = x_card + inset
            img_y1 = y_card + inset
            img_w  = card_w  - 2 * inset
            img_h  = card_h  - 2 * inset
            big    = cv2.resize(thumb, (img_w, img_h))
            frame[img_y1:img_y1 + img_h, img_x1:img_x1 + img_w] = big

            # ── Top-left numbered badge ───────────────────────────────────
            badge_r = 18
            badge_cx = x_card + inset + badge_r + 4
            badge_cy = y_card + inset + badge_r + 4
            cv2.circle(frame, (badge_cx, badge_cy), badge_r, ac, -1)
            cv2.putText(frame, str(i + 1),
                        (badge_cx - 7, badge_cy + 7),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75,
                        (10, 5, 20), 2, cv2.LINE_AA)

            # ── Label below card ─────────────────────────────────────────
            label_y = y_card + card_h + 28
            label_text = ["SKETCH  ONE", "SKETCH  TWO", "SKETCH  THREE"][i]
            text_size = cv2.getTextSize(label_text,
                                        cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)[0]
            label_x = x_card + (card_w - text_size[0]) // 2
            cv2.putText(frame, label_text,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65,
                        ac, 1, cv2.LINE_AA)

            # Thin underline
            cv2.line(frame,
                     (label_x, label_y + 5),
                     (label_x + text_size[0], label_y + 5),
                     ac, 1)

        return frame


# ──────────────────────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────────────────────

class VirtualPaint:
    """
    Main application class that orchestrates:
      - Camera capture
      - Hand detection
      - Gesture interpretation
      - Canvas drawing
      - Gallery management
    """

    MODE_IDLE     = "IDLE"
    MODE_DRAW     = "DRAWING"
    MODE_SELECT   = "SELECTION"

    def __init__(self):
        self.cap        = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # flush stale frames → lower latency

        self.detector    = HandDetector()
        self.toolbar     = Toolbar(FRAME_W, HEADER_H)
        self.gallery     = Gallery()

        # Drawing state
        self.canvas      = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.prev_pt     = None
        self.mode        = self.MODE_IDLE
        self.colour      = "Red"
        self.brush       = BRUSH_SIZES[1]

        # EWMA smoothed position (exponential weighted moving average)
        self._smooth_pt: tuple[float, float] | None = None

        # Gesture history for majority-vote stabilisation
        from collections import deque
        self._gesture_hist: deque = deque(maxlen=GESTURE_HISTORY)

        # FPS bookkeeping
        self._fps_time   = time.time()
        self._fps        = 0
        self._frame_cnt  = 0

        # Fist-save debounce
        self._fist_frames  = 0
        self.FIST_THRESH   = FIST_THRESH
        self._save_cooldown = 0   # frames remaining where gestures are ignored

        # Gallery sidebar width
        self._side_w = Gallery.PANEL_W

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _smooth(self, pt) -> tuple[int, int]:
        """Exponential Weighted Moving Average — low lag, smooth cursor."""
        α = SMOOTHING
        if self._smooth_pt is None:
            self._smooth_pt = (float(pt[0]), float(pt[1]))
        else:
            sx, sy = self._smooth_pt
            self._smooth_pt = (α * pt[0] + (1 - α) * sx,
                               α * pt[1] + (1 - α) * sy)
        return int(self._smooth_pt[0]), int(self._smooth_pt[1])

    def _stable_gesture(self, raw: str) -> str:
        """Add raw gesture to history and return the majority-vote winner."""
        self._gesture_hist.append(raw)
        return max(set(self._gesture_hist), key=self._gesture_hist.count)

    def _update_fps(self):
        self._frame_cnt += 1
        now = time.time()
        if now - self._fps_time >= 0.5:
            self._fps = self._frame_cnt / (now - self._fps_time)
            self._fps_time = now
            self._frame_cnt = 0

    def _draw_hud(self, frame: np.ndarray):
        """Overlay FPS, mode badge, and colour indicator."""
        # FPS
        cv2.putText(frame, f"FPS: {self._fps:.1f}",
                    (10, FRAME_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)

        # Mode badge
        colour_map = {
            self.MODE_DRAW   : (0, 220, 0),
            self.MODE_SELECT : (0, 180, 255),
            self.MODE_IDLE   : (80, 80, 80),
        }
        badge_col = colour_map.get(self.mode, (80, 80, 80))
        cv2.putText(frame, f"Mode: {self.mode}",
                    (10, FRAME_H - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_col, 2, cv2.LINE_AA)

        # Active colour dot
        dot_x = 200
        cv2.circle(frame, (dot_x, FRAME_H - 27), 14, COLOURS[self.colour], -1)
        cv2.circle(frame, (dot_x, FRAME_H - 27), 14, (255, 255, 255), 1)

        # Gesture hints (bottom right)
        hints = [
            "☝  Index only  → Draw",
            "✌  Two fingers → Select tool",
            "✊  Fist (hold) → Save & clear",
            "Q              → Quit",
        ]
        for i, h in enumerate(hints):
            cv2.putText(frame, h,
                        (FRAME_W - self._side_w - 420,
                         FRAME_H - 15 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (160, 160, 160), 1, cv2.LINE_AA)

    # ── Gesture Interpretation ─────────────────────────────────────────────────

    def _interpret_gesture(self, fingers: list[int]) -> str:
        """
        Map a fingers-up pattern to a named gesture.

          [Thumb, Index, Middle, Ring, Pinky]
          Drawing  : [_, 1, 0, 0, _]   — index only
          Selection: [_, 1, 1, 0, _]   — index + middle
          Fist     : [0, 0, 0, 0, 0]   — all down
        """
        _, i, m, r, p = fingers
        if i == 1 and m == 0 and r == 0 and p == 0:
            return self.MODE_DRAW
        if i == 1 and m == 1 and r == 0:
            return self.MODE_SELECT
        if sum(fingers) == 0:
            return "FIST"
        return self.MODE_IDLE

    # ── Core Loop ─────────────────────────────────────────────────────────────

    def run(self):
        print("\n[INFO] Camera started. Gesture guide:")
        print("       ☝  Index finger up            → Draw")
        print("       ✌  Index + Middle up          → Selection mode")
        print("       ✊  Close fist (hold ~18 fr)   → Save & clear canvas")
        print("       Q  (keyboard)                  → Quit\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Frame capture failed — retrying…")
                continue

            frame = cv2.flip(frame, 1)          # mirror view
            frame = self.detector.process(frame)
            fingers = self.detector.fingers_up()
            tip_raw = self.detector.index_tip
            self._update_fps()

            raw_gesture = self._interpret_gesture(fingers)
            gesture     = self._stable_gesture(raw_gesture)  # majority-vote

            # ── Post-save cooldown: ignore gestures briefly after saving ──────
            if self._save_cooldown > 0:
                self._save_cooldown -= 1
                gesture = self.MODE_IDLE

            # ── Fist debounce: must hold for FIST_THRESH frames ──────────────
            if gesture == "FIST":
                self._fist_frames += 1
                # Progress ring around palm centre
                if len(self.detector.landmarks) > 9:
                    cx, cy = self.detector.landmarks[9]  # palm centre (MCP mid)
                else:
                    cx, cy = FRAME_W // 2, FRAME_H // 2
                pct   = min(self._fist_frames / self.FIST_THRESH, 1.0)
                angle = int(360 * pct)
                cv2.ellipse(frame, (cx, cy), (40, 40), -90, 0, angle,
                            (50, 220, 50), 4)
                # Countdown label
                frames_left = self.FIST_THRESH - self._fist_frames
                cv2.putText(frame, f"Hold… {max(frames_left,0)}",
                            (cx - 45, cy + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2, cv2.LINE_AA)

                if self._fist_frames >= self.FIST_THRESH:
                    self._save_canvas()
                    self._fist_frames  = 0
                    self._save_cooldown = SAVE_COOLDOWN
            else:
                self._fist_frames = 0

            # ── Mode switching ────────────────────────────────────────────────
            self.mode = gesture if gesture != "FIST" else self.MODE_IDLE

            # ── Selection mode: toolbar hit-test ─────────────────────────────
            if self.mode == self.MODE_SELECT and tip_raw:
                tip_sm = self._smooth(tip_raw)
                hit = self.toolbar.hit_test(tip_sm)
                if hit:
                    if hit["type"] == "colour":
                        self.colour = hit["value"]
                    elif hit["type"] == "brush":
                        self.brush = hit["value"]
                # Visual cursor
                cv2.circle(frame, tip_sm, 12, (0, 200, 255), 2)
                self.prev_pt = None           # don't connect dots across modes

            # ── Drawing mode ──────────────────────────────────────────────────
            elif self.mode == self.MODE_DRAW and tip_raw:

                # ── ERASER: palm-area wipe ────────────────────────────────────
                if self.colour == "Eraser":
                    hull = self.detector.palm_hull()
                    if hull is not None:
                        # Only erase below the toolbar
                        # Clamp hull points so they don't touch the toolbar area
                        hull[:, :, 1] = np.maximum(hull[:, :, 1], HEADER_H)

                        # Erase on canvas (set pixels inside hull to black)
                        cv2.fillConvexPoly(self.canvas, hull, (0, 0, 0))

                        # Visual feedback: semi-transparent red overlay
                        overlay = frame.copy()
                        cv2.fillConvexPoly(overlay, hull, (30, 30, 220))
                        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

                        # Glowing border around erase region
                        cv2.polylines(frame, [hull], True, (0, 80, 255), 2,
                                      cv2.LINE_AA)

                        # "ERASING" label near palm centre
                        M = cv2.moments(hull)
                        if M["m00"] != 0:
                            palm_cx = int(M["m10"] / M["m00"])
                            palm_cy = int(M["m01"] / M["m00"])
                            cv2.putText(frame, "ERASING",
                                        (palm_cx - 40, palm_cy + 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 100, 255), 2, cv2.LINE_AA)
                    self.prev_pt = None

                # ── NORMAL PEN: single-point draw ─────────────────────────────
                else:
                    tip_sm = self._smooth(tip_raw)
                    # Only draw below the toolbar
                    if tip_sm[1] > HEADER_H:
                        if self.prev_pt and self.prev_pt[1] > HEADER_H:
                            cv2.line(self.canvas,
                                     self.prev_pt, tip_sm,
                                     COLOURS[self.colour],
                                     self.brush, cv2.LINE_AA)
                        self.prev_pt = tip_sm
                        # Draw cursor dot
                        cv2.circle(frame, tip_sm, self.brush // 2 + 2,
                                   COLOURS[self.colour], -1)
                    else:
                        self.prev_pt = None

            else:
                self.prev_pt    = None
                self._smooth_pt = None   # reset EWMA so cursor doesn't drag

            # ── Merge canvas onto camera frame ────────────────────────────────
            canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            frame = cv2.add(frame_bg, canvas_fg)

            # ── Toolbar & sidebar ─────────────────────────────────────────────
            self.toolbar.draw(frame, self.colour, self.brush, len(self.gallery.items))
            if self.gallery.items:
                self.gallery.draw_sidebar(frame)

            # ── HUD ───────────────────────────────────────────────────────────
            self._draw_hud(frame)

            cv2.imshow("Virtual Paint", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] User pressed Q — exiting.")
                break
            elif key == ord("s"):          # keyboard shortcut as fallback
                self._save_canvas()

        self._cleanup()

    # ── Save & Cleanup ────────────────────────────────────────────────────────

    def _save_canvas(self):
        """Save the current canvas as a gallery item and reset."""
        if len(self.gallery.items) >= MAX_ITEMS:
            print("[INFO] Max items reached — cannot save more.")
            return

        # Only save if there is something on the canvas
        if not np.any(self.canvas):
            print("[WARN] Canvas is empty — nothing to save.")
            return

        self.gallery.add(self.canvas.copy())
        n = len(self.gallery.items)
        print(f"[INFO] Saved drawing #{n}/{MAX_ITEMS}")

        # Flash confirmation
        flash = self.canvas.copy()
        cv2.putText(flash, f"Saved #{n}!", (FRAME_W // 2 - 80, FRAME_H // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        # Show frame with flash overlay
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.addWeighted(frame, 0.5, flash, 0.5, 0, frame)
            cv2.imshow("Virtual Paint", frame)
            cv2.waitKey(400)

        self.canvas[:] = 0           # clear canvas
        self.prev_pt = None

        if n >= MAX_ITEMS:
            print(f"\n[INFO] All {MAX_ITEMS} drawings captured! Showing gallery…")
            self._show_gallery_in_window()
            self._cleanup()
            sys.exit(0)

    def _show_gallery_in_window(self):
        """Render the premium gallery inside the SAME 'Virtual Paint' window."""
        gallery_frame = self.gallery.build_gallery_frame()
        while True:
            cv2.imshow("Virtual Paint", gallery_frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):   # Q or Esc to exit
                break

    def _cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Goodbye!")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if not ask_camera_permission():
        print("[INFO] Camera access denied by user. Exiting.")
        sys.exit(0)

    print("[INFO] Camera access granted. Starting Virtual Paint…")
    try:
        app = VirtualPaint()
        app.run()
    except RuntimeError as err:
        print(f"[ERROR] {err}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")


if __name__ == "__main__":
    main()
