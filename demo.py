"""
draw.py — Visualization demo for the from-scratch neural network.

This UI is a showcase layer to demonstrate the trained model running live.
The neural network itself is implemented from scratch in neural_network.py;
this file only provides a drawing canvas and feeds input to the real model.
"""

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
from neural_network import NeuralNetwork

# --- load the trained model ---
nn = NeuralNetwork()                      # fresh network (random weights)
data = np.load('trained_model.npz')       # load the weights you trained
nn.w1, nn.b1 = data['w1'], data['b1']     # overwrite with trained values
nn.w2, nn.b2 = data['w2'], data['b2']

# --- drawing setup ---
CANVAS_SIZE = 280        # big canvas for comfortable mouse drawing
BRUSH = 12               # brush thickness

class App:
    def __init__(self, root):
        self.root = root
        root.title("Draw a digit (0-9)")

        # canvas the user draws on
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)   # drag = draw

        # a parallel PIL image we draw onto (so we can resize it to 28x28)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # buttons
        btns = tk.Frame(root)
        btns.pack(pady=5)
        tk.Button(btns, text="Predict", command=self.predict).pack(side="left", padx=5)
        tk.Button(btns, text="Clear", command=self.clear).pack(side="left", padx=5)

        # label to show the result
        self.result = tk.Label(root, text="Draw a digit, then hit Predict",
                               font=("Arial", 18))
        self.result.pack(pady=10)

    def paint(self, event):
        # draw a white blob at the cursor on BOTH the visible canvas and the PIL image
        x, y = event.x, event.y
        r = BRUSH
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result.config(text="Draw a digit, then hit Predict")

    def predict(self):
        # convert the drawing to a numpy array
        arr = np.array(self.image, dtype=np.float64)

        # if nothing is drawn, bail out
        if arr.sum() == 0:
            self.result.config(text="Draw something first!")
            return

        # --- center the digit by its center of mass ---
        # find the centroid (weighted average position of the bright pixels)
        rows = np.arange(arr.shape[0])
        cols = np.arange(arr.shape[1])
        total = arr.sum()
        center_row = (arr.sum(axis=1) @ rows) / total
        center_col = (arr.sum(axis=0) @ cols) / total

        # how far the centroid is from the true center
        shift_row = int(arr.shape[0] / 2 - center_row)
        shift_col = int(arr.shape[1] / 2 - center_col)

        # shift the image so the digit is centered
        arr = np.roll(arr, shift_row, axis=0)
        arr = np.roll(arr, shift_col, axis=1)

        # --- resize to 28x28, normalize, flatten ---
        centered_img = Image.fromarray(arr.astype(np.uint8))
        small = centered_img.resize((28, 28))
        pixels = np.array(small) / 255.0
        x = pixels.reshape(784)

        # run the REAL network's forward pass
        a2, _, _ = nn.forward(x)
        digit = np.argmax(a2)
        confidence = a2[digit]
        self.result.config(text=f"Prediction: {digit}   ({confidence*100:.1f}%)")

root = tk.Tk()
App(root)
root.mainloop()


