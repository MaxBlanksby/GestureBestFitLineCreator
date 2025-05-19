# ✋📈 Hand-Drawn Function Fitter

This project uses your **webcam** and **hand gestures** to draw functions in real time. As you draw in the air using your **index finger**, the program collects points and fits them to:

- 📉 A linear (best-fit) line  
- 📐 A 3rd-degree polynomial  
- 🔁 A sinusoidal function

You can **clear the graph** by making a ✊ **fist**, and toggle to see **only the best-fit curve** (lowest error) using ✌️ **scissors**.

---

## 🔧 How It Works

- Built with `OpenCV` + `Mediapipe` for real-time hand tracking.
- Uses `NumPy`, `SciPy`, and `Matplotlib` to compute and visualize fits.
- Equation labels are updated **live** as more data is drawn.
- The program intelligently decides the **most accurate model** (lowest mean squared error).

---

## 🎮 Controls

| Gesture  | Action                  |
|----------|-------------------------|
| ☝️ Point | Draw on graph           |
| ✊ Rock  | Clear screen             |
| ✌️ Scissors | Toggle to best-fit only |

---


