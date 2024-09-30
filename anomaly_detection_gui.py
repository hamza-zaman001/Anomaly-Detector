import numpy as np
import random
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from collections import deque
import threading

# Simulate Data Stream with random patterns and seasonal variations
def data_stream_generator(num_points=1000, anomaly_ratio=0.05):
    """
    Generates a continuous data stream with a mix of regular, seasonal, and random values.
    A portion of the data is marked as anomalies.

    Args:
    - num_points: Total number of data points to generate.
    - anomaly_ratio: The proportion of anomalies in the data stream.

    Returns:
    - A generator that yields values from the data stream one by one.
    """
    data = []
    for i in range(num_points):
        base_value = 100 + random.uniform(-10, 10)  # Random normal variation
        seasonal_effect = i % 100  # Seasonal pattern
        value = base_value + seasonal_effect
        
        if random.random() < anomaly_ratio:  # Introduce anomalies at random
            value += random.uniform(50, 100)  # Create spike anomalies
        
        data.append(value)

    for value in data:
        yield value
        time.sleep(0.01)  # Simulate a real-time data stream delay

# Anomaly Detection using Isolation Forest
class AnomalyDetector:
    def __init__(self, window_size=100):
        """
        Initializes the anomaly detector with a sliding window to detect anomalies in real-time.
        
        Args:
        - window_size: Number of recent points considered for anomaly detection at a time.
        """
        self.model = IsolationForest(contamination=0.05)  # Assumes 5% anomalies
        self.window_size = window_size
        self.data_window = deque(maxlen=window_size)
        self.is_model_trained = False

    def detect(self, new_value):
        """
        Detects anomalies using a sliding window approach and Isolation Forest algorithm.
        
        Args:
        - new_value: The latest value in the data stream.
        
        Returns:
        - A boolean indicating if the new_value is an anomaly.
        """
        self.data_window.append([new_value])  # Append new value to window
        
        # Train model once window is full to avoid excessive retraining
        if len(self.data_window) >= self.window_size:
            if not self.is_model_trained:
                self.model.fit(self.data_window)  # Train on current window
                self.is_model_trained = True
            prediction = self.model.predict([[new_value]])  # Predict the latest value
            return prediction[0] == -1  # Return True if anomaly detected
        
        return False  # If not enough data, assume no anomaly

    def retrain_model(self):
        """
        Retrains the model dynamically if the detection rate seems off.
        This can be hooked to sensitivity adjustments in the GUI.
        """
        if len(self.data_window) >= self.window_size:
            self.model.fit(self.data_window)
            self.is_model_trained = True

# Real-time GUI for anomaly detection visualization
class RealTimeGUI:
    def __init__(self, root):
        """
        Initializes the GUI for real-time anomaly detection visualization.
        
        Args:
        - root: The Tkinter root window.
        """
        self.root = root
        self.root.title("Real-Time Anomaly Detection")
        
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.data_points = []
        self.anomaly_flags = []
        self.detector = AnomalyDetector()
        self.data_generator = data_stream_generator()
        self.running = False

        # Start/Stop buttons
        self.start_button = tk.Button(root, text="Start", command=self.start_stream)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_stream)
        self.stop_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Sensitivity Adjustment Scale (dynamic retraining)
        self.sensitivity_scale = tk.Scale(root, from_=0.01, to=0.5, resolution=0.01, 
                                          label="Detection Sensitivity", orient="horizontal", command=self.update_sensitivity)
        self.sensitivity_scale.set(0.05)  # Default 5% anomaly detection
        self.sensitivity_scale.pack(side=tk.BOTTOM, padx=10, pady=10)

    def update_plot(self):
        """
        Updates the plot with new data points and highlights anomalies.
        """
        self.ax.clear()
        self.ax.plot(self.data_points, label='Data Stream', color='blue')
        
        # Highlight anomalies
        anomalies = [self.data_points[i] if self.anomaly_flags[i] else np.nan for i in range(len(self.data_points))]
        self.ax.scatter(range(len(self.data_points)), anomalies, color='red', label='Anomalies')
        
        self.ax.set_title('Efficient Data Stream Anomaly Detection')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Data Value')
        self.ax.legend()
        
        self.canvas.draw()

    def data_stream(self):
        """
        Handles the real-time data stream, updating the plot and detecting anomalies.
        """
        while self.running:
            try:
                new_value = next(self.data_generator)
                is_anomaly = self.detector.detect(new_value)
                
                self.data_points.append(new_value)
                self.anomaly_flags.append(is_anomaly)

                if len(self.data_points) > 200:  # Limit display to the last 200 points
                    self.data_points.pop(0)
                    self.anomaly_flags.pop(0)
                
                self.update_plot()  # Update the plot with new data

                time.sleep(0.05)  # Adjust update speed if necessary
            except StopIteration:
                break

    def update_sensitivity(self, val):
        """
        Updates the detection sensitivity based on user input.
        Retrains the model dynamically with the new sensitivity.
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

# Main function to initialize the Tkinter GUI
def main():
    root = tk.Tk()
    app = RealTimeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
