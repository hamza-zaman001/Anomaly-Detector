import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time
from anomaly_detector import AnomalyDetector
from data_stream import data_stream_generator
import numpy as np
import sys  # For clean exit

class RealTimeGUI:
    def __init__(self, root):
        """
        Initializes the real-time visualization GUI for anomaly detection.
        
        Args:
        - root: The Tkinter root window.
        """
        self.root = root
        self.root.title("Real-Time Anomaly Detection Dashboard")
        self.root.geometry("900x650")
        self.root.configure(bg="#2b2d42")  # Modern dark background

        # Set up the Matplotlib figure and Tkinter canvas for plotting
        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=20, pady=20)

        # Stores the data stream and anomaly flags
        self.data_points = []
        self.anomaly_flags = []

        # Anomaly detector and data generator
        self.detector = AnomalyDetector()
        self.data_generator = data_stream_generator()
        self.running = False

        # Control frame for buttons and scale
        control_frame = tk.Frame(self.root, bg="#2b2d42")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

        # Start/Stop buttons
        self.start_button = tk.Button(control_frame, text="Start Stream", font=("Arial", 12), 
                                      command=self.start_stream, bg="#4CAF50", fg="white", width=12)
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        self.stop_button = tk.Button(control_frame, text="Stop Stream", font=("Arial", 12), 
                                     command=self.stop_stream, bg="#F44336", fg="white", width=12, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=10, pady=10)

        # Sensitivity adjustment scale with labels
        self.sensitivity_label = tk.Label(control_frame, text="Detection Sensitivity", 
                                          font=("Arial", 12), bg="#2b2d42", fg="#edf2f4")
        self.sensitivity_label.grid(row=0, column=2, padx=10)

        self.sensitivity_scale = tk.Scale(control_frame, from_=0.01, to=0.5, resolution=0.01, 
                                          orient="horizontal", command=self.update_sensitivity, 
                                          bg="#2b2d42", fg="#edf2f4", highlightbackground="#2b2d42")
        self.sensitivity_scale.set(0.05)
        self.sensitivity_scale.grid(row=0, column=3, padx=10)

        # Status label to show running status
        self.status_label = tk.Label(self.root, text="Status: Idle", font=("Arial", 14), bg="#2b2d42", fg="#edf2f4")
        self.status_label.pack(side=tk.BOTTOM, pady=10)

        # Bind close event to properly exit the program
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_plot(self):
        """
        Updates the plot with the latest data and highlights anomalies.
        """
        self.ax.clear()
        self.ax.plot(self.data_points, label='Data Stream', color='#8ecae6', linewidth=2)
        
        # Highlight anomalies in red
        anomalies = [self.data_points[i] if self.anomaly_flags[i] else np.nan for i in range(len(self.data_points))]
        self.ax.scatter(range(len(self.data_points)), anomalies, color='#ff3b30', label='Anomalies', zorder=5)
        
        self.ax.set_title('Real-Time Anomaly Detection', fontsize=16, color="#2b2d42")
        self.ax.set_xlabel('Time', fontsize=12, color="#2b2d42")
        self.ax.set_ylabel('Data Value', fontsize=12, color="#2b2d42")
        self.ax.legend()
        self.ax.grid(True, color='#8d99ae', linestyle='--', linewidth=0.5)
        
        self.canvas.draw()

    def data_stream(self):
        """
        Streams data in real-time and detects anomalies as they arrive.
        """
        self.status_label.config(text="Status: Streaming...", fg="#00bfae")
        while self.running:
            try:
                new_value = next(self.data_generator)
                is_anomaly = self.detector.detect(new_value)
                
                self.data_points.append(new_value)
                self.anomaly_flags.append(is_anomaly)

                if len(self.data_points) > 200:
                    self.data_points.pop(0)
                    self.anomaly_flags.pop(0)
                
                self.update_plot()
                time.sleep(0.05)
            except StopIteration:
                break
        self.status_label.config(text="Status: Idle", fg="#edf2f4")

    def update_sensitivity(self, val):
        """
        Updates the anomaly detection sensitivity based on the user's input.
        """
        new_sensitivity = float(val)
        self.detector.model.set_params(contamination=new_sensitivity)
        self.detector.retrain_model()

    def start_stream(self):
        """
        Starts the data stream in a separate thread.
        """
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=self.data_stream, daemon=True).start()

    def stop_stream(self):
        """
        Stops the data stream.
        """
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped", fg="#ff3b30")

    def on_closing(self):
        """
        Handles the window close event and cleans up the program.
        """
        self.stop_stream()  # Stop the stream if it's running
        self.root.quit()  # Close the Tkinter window
        self.root.destroy()  # Destroy the Tkinter mainloop
