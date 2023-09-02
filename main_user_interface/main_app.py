import tkinter as tk
from tkinter import *
from tkinter import ttk

def start_button_clicked():
    print("Start button clicked!")

# Create the main window
root = tk.Tk()
root.title("Simple UI Frame")

# Determine screen dimensions
screen_width = 360
screen_height = 690

# Calculate centered position
x_position = (root.winfo_screenwidth() - screen_width) // 2
y_position = (root.winfo_screenheight() - screen_height) // 2

# Set window size and position
root.geometry(f"{screen_width}x{screen_height}+{x_position}+{y_position}")

# Create a frame with padding
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(fill=tk.BOTH, expand=True)

# Create a button in the middle of the frame
start_button = tk.Button(frame, text="Start", command=start_button_clicked)
start_button.pack(side=tk.TOP, pady=100)

# Start the Tkinter main loop
root.mainloop()
