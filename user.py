import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import pyttsx3
import json
import os

# ==== YOUR FILE PATHS ====
MODEL_PATH = r"C:\Users\sinch\OneDrive\Desktop\dcproject\char_digit_model.h5"
LABEL_MAP_PATH = r"C:\Users\sinch\OneDrive\Desktop\dcproject\label_map.json"
IMG_SIZE = (64, 64)
PREVIEW_SIZE = (280, 280)

DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three",
    "4": "four", "5": "five", "6": "six",
    "7": "seven", "8": "eight", "9": "nine"
}


class PredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Digit & Character Recognition")
        self.root.geometry("500x650")
        self.root.configure(bg="#F2F3F4")
        self.root.resizable(False, False)

        # Load model & label map
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            raise

        try:
            with open(LABEL_MAP_PATH, "r") as f:
                label_map = json.load(f)
            self.label_map = {int(k): v for k, v in label_map.items()}
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load label map:\n{e}")
            raise

        self.build_ui()

        self.selected_path = None
        self.preview_img = None

        self.tts_engine = pyttsx3.init()

    # ---- NEW UI ----
    def build_ui(self):
        title = tk.Label(
            self.root, 
            text="Handwritten Digit & Character Predictor",
            font=("Segoe UI", 17, "bold"),
            bg="#F2F3F4",
            fg="#2C3E50"
        )
        title.pack(pady=15)

        # Image preview frame
        self.image_frame = tk.Frame(self.root, width=300, height=300, bg="white", bd=2, relief="ridge")
        self.image_frame.pack(pady=10)
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack()

        # Styled Buttons
        self.select_btn = tk.Button(
            self.root, text="📁  Select Image",
            command=self.select_image,
            font=("Segoe UI", 12, "bold"),
            bg="#3498DB", fg="white",
            activebackground="#2980B9",
            relief="flat", width=20, height=1
        )
        self.select_btn.pack(pady=10)

        self.predict_btn = tk.Button(
            self.root, text="🔊  Predict & Speak",
            command=self.predict_and_speak,
            font=("Segoe UI", 12, "bold"),
            bg="#27AE60", fg="white",
            activebackground="#1E8449",
            relief="flat", width=20, height=1,
            state="disabled"
        )
        self.predict_btn.pack(pady=5)

        # Prediction Result
        self.result_label = tk.Label(
            self.root,
            text="Prediction: —",
            font=("Segoe UI", 15),
            bg="#F2F3F4", fg="#2C3E50"
        )
        self.result_label.pack(pady=20)

        self.conf_label = tk.Label(
            self.root,
            text="Confidence: —",
            font=("Segoe UI", 12),
            bg="#F2F3F4", fg="#2C3E50"
        )
        self.conf_label.pack()

    # -----------------------------

    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", ".png;.jpg;.jpeg;.bmp")]
        )
        if not path:
            return

        self.selected_path = path

        img = Image.open(path).convert("RGB")
        img.thumbnail(PREVIEW_SIZE)
        self.preview_img = ImageTk.PhotoImage(img)

        self.image_label.configure(image=self.preview_img)
        self.predict_btn.configure(state="normal")

        self.result_label.configure(text="Prediction: —")
        self.conf_label.configure(text="Confidence: —")

    # -----------------------------

    def preprocess(self, path):
        img = Image.open(path).convert("L")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype="float32")

        arr = np.expand_dims(arr, axis=-1)
        arr = np.expand_dims(arr, axis=0)
        return arr

    # -----------------------------

    def predict_and_speak(self):
        if not self.selected_path:
            return

        x = self.preprocess(self.selected_path)
        preds = self.model.predict(x)
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])
        pred_char = self.label_map[idx]

        self.result_label.configure(text=f"Prediction: {pred_char}")
        self.conf_label.configure(text=f"Confidence: {confidence*100:.2f}%")

        # Speak
        if pred_char in DIGIT_WORDS:
            speech = f"The predicted number is {DIGIT_WORDS[pred_char]}."
        else:
            speech = f"The predicted character is {pred_char}."
        self.tts_engine.say(speech)
        self.tts_engine.runAndWait()


def main():
    root = tk.Tk()
    PredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()