
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

def sine_func(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

FINGERS = [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
           mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
           mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
           mp.solutions.hands.HandLandmark.PINKY_TIP]
PIPS   = [mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
           mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
           mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
           mp.solutions.hands.HandLandmark.PINKY_PIP]

def get_finger_states(hand_landmarks):
    return [hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
            for tip, pip in zip(FINGERS, PIPS)]

def classify_gesture(states):
    if all(not s for s in states):
        return "Rock"
    if states == [True, False, False, False]:
        return "Point"
    if states[0] and states[1] and not states[2] and not states[3]:
        return "Scissors"
    if all(states):
        return "Paper"
    return "Unknown"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

plt.ion()
fig, ax = plt.subplots()
pts_plot, = ax.plot([], [], 'ro', label='Drawn Points')
line_plot, = ax.plot([], [], 'y-', label='Linear Fit')
poly_plot, = ax.plot([], [], 'b--', label='Cubic Fit')
sine_plot, = ax.plot([], [], 'g:', label='Sinusoidal Fit')
# Equation text
line_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, va='top')
poly_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, va='top')
sine_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, va='top')
ax.set_title('Hand-Drawn Function Fitter')
ax.set_xlabel('Frame Index')
ax.set_ylabel('Vertical Position')
ax.legend(loc='lower right')

points = []
frame_idx = 0
show_best_mode = False
last_gesture = None

def update_fits(points, best_mode):
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    m, b = np.polyfit(xs, ys, 1)
    y_line = m*xs + b
    err_line = np.mean((y_line - ys)**2)
    c3, c2, c1, c0 = np.polyfit(xs, ys, 3)
    poly_fn = np.poly1d([c3, c2, c1, c0])
    y_poly = poly_fn(xs)
    err_poly = np.mean((y_poly - ys)**2)
    try:
        guess = [(ys.max()-ys.min())/2, 2*np.pi/(xs.max()-xs.min()), 0, ys.mean()]
        popt, _ = curve_fit(sine_func, xs, ys, p0=guess, maxfev=10000)
        y_sine = sine_func(xs, *popt)
        err_sine = np.mean((y_sine - ys)**2)
    except Exception:
        err_sine = np.inf

    errors = {'Linear': err_line, 'Cubic': err_poly, 'Sinusoidal': err_sine}
    best_model = min(errors, key=errors.get)

    x_line = np.linspace(xs.min(), xs.max(), 200)
    line_data = m*x_line + b
    poly_data = poly_fn(x_line)
    sine_data = sine_func(x_line, *popt) if err_sine < np.inf else None

    pts_plot.set_data(xs, ys)
    line_plot.set_data([], [])
    poly_plot.set_data([], [])
    sine_plot.set_data([], [])
    line_text.set_text('')
    poly_text.set_text('')
    sine_text.set_text('')

    if best_mode:
        if best_model == 'Linear':
            line_plot.set_data(x_line, line_data)
            line_text.set_text(f"Best Linear: y={m:.3f}x+{b:.3f}")
        elif best_model == 'Cubic':
            poly_plot.set_data(x_line, poly_data)
            poly_text.set_text(f"Best Cubic: y={c3:.3f}x³+{c2:.3f}x²+{c1:.3f}x+{c0:.3f}")
        else:
            sine_plot.set_data(x_line, sine_data)
            A,B,C,D = popt
            sine_text.set_text(f"Best Sine: y={A:.3f}sin({B:.3f}x+{C:.3f})+{D:.3f}")
    else:
        line_plot.set_data(x_line, line_data)
        poly_plot.set_data(x_line, poly_data)
        if sine_data is not None:
            sine_plot.set_data(x_line, sine_data)
        line_text.set_text(f"Linear: y={m:.3f}x+{b:.3f}")
        poly_text.set_text(f"Cubic: y={c3:.3f}x³+{c2:.3f}x²+{c1:.3f}x+{c0:.3f}")
        if sine_data is not None:
            A,B,C,D = popt
            sine_text.set_text(f"Sine: y={A:.3f}sin({B:.3f}x+{C:.3f})+{D:.3f}")

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw(); fig.canvas.flush_events()

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    gesture = 'None'
    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        states = get_finger_states(hand)
        gesture = classify_gesture(states)
        if gesture == 'Scissors' and last_gesture != 'Scissors':
            show_best_mode = not show_best_mode
        if gesture == 'Rock':
            points.clear(); frame_idx = 0
            pts_plot.set_data([], []); line_plot.set_data([], []);
            poly_plot.set_data([], []); sine_plot.set_data([], [])
            line_text.set_text(''); poly_text.set_text(''); sine_text.set_text('')
        elif gesture == 'Point':
            lm = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_px = int(lm.x * frame.shape[1]); y_px = int(lm.y * frame.shape[0])
            points.append((frame_idx, frame.shape[0]-y_px)); frame_idx += 1

    cv2.putText(frame, f"Gesture: {gesture}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,255), 2, cv2.LINE_AA)

    if res.multi_hand_landmarks and len(points) >= 5:
        update_fits(points, show_best_mode)

    cv2.imshow('Hand Drawing', frame)
    time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    last_gesture = gesture

cap.release(); cv2.destroyAllWindows()
