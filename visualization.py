import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time
from anomaly_detector import AnomalyDetector
from data_stream import data_stream_generator
import numpy as np

class RealTimeGUI:
    def __init__(self, root):
        """
        Initializes the real-time visualization GUI for anomaly detection.
        
        Args:
        - root: The Tkinter root window.
        """
        self.root = root
        self.root.title("Real-Time Anomaly Detection")
        
        # Set up the Matplotlib figure and Tkinter canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.data_points = []  # Stores the data stream
        self.anomaly_flags = []  # Flags for whether each point is an anomaly
        self.detector = AnomalyDetector()  # Create an instance of the anomaly detector
        self.data_generator = data_stream_generator()  # Initialize the data generator
        self.running = False  # Controls the state of the stream
        
        # Start/Stop buttons
        self.start_button = tk.Button(root, text="Start", command=self.start_stream)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_stream)
        self.stop_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Sensitivity adjustment scale
        self.sensitivity_scale = tk.Scale(root, from_=0.01, to=0.5, resolution=0.01, 
                                          label="Detection Sensitivity", orient="horizontal", command=self.update_sensitivity)
        self.sensitivity_scale.set(0.05)  # Default sensitivity
        self.sensitivity_scale.pack(side=tk.BOTTOM, padx=10, pady=10)

    def update_plot(self):
        """
        Updates the plot with the latest data and highlights anomalies.
        """
        self.ax.clear()
        self.ax.plot(self.data_points, label='Data Stream', color='blue')
        
        # Highlight anomalies in red
        anomalies = [self.data_points[i] if self.anomaly_flags[i] else np.nan for i in range(len(self.data_points))]
        self.ax.scatter(range(len(self.data_points)), anomalies, color='red', label='Anomalies')
        
        self.ax.set_title('Efficient Data Stream Anomaly Detection')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Data Value')
        self.ax.legend()
        
        self.canvas.draw()

    def data_stream(self):
        """
        Streams data in real-time and detects anomalies as they arrive.
        """
        while self.running:
            try:
                new_value = next(self.data_generator)  # Get the next value from the data stream
                is_anomaly = self.detector.detect(new_value)  # Check if it's an anomaly
                
                self.data_points.append(new_value)
                self.anomaly_flags.append(is_anomaly)

                # Limit the graph display to the last 200 points
                if len(self.data_points) > 200:
                    self.data_points.pop(0)
                    self.anomaly_flags.pop(0)
                
                self.update_plot()  # Update the plot with the latest data
                
                time.sleep(0.05)  # Control the speed of updates
            except StopIteration:
                break

    def update_sensitivity(self, val):
        """
        Updates the anomaly detection sensitivity based on the user's input.
        Retrains the model with the new sensitivity.
        """
        new_sensitivity = float(val)
        self.detector.model.set_params(contamination=new_sensitivity)
        self.detector.retrain_model()

    def start_stream(self):
        """
        Starts the data stream in a separate thread.
        """
        self.running = True
        threading.Thread(target=self.data_stream, daemon=True).start()

    def stop_stream(self):
        """
        Stops the data stream.
        """
        self.running = False
