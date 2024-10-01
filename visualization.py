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

        # Initialize line plot for the data stream
        self.line, = self.ax.plot([], [], label='Data Stream', color='#8ecae6', linewidth=2)
        
        # Initialize scatter plot for anomalies
        self.anomaly_scatter = self.ax.scatter([], [], color='#ff3b30', label='Anomalies', zorder=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=20, pady=20)

        # Initialize data points and anomaly flags
        self.data_points = []
        self.anomaly_flags = []

        # Anomaly detector and data generator
        self.detector = AnomalyDetector()
        self.data_generator = data_stream_generator()
        self.running = False

        # Control frame for buttons and scale
        control_frame = tk.Frame(self.root, bg="#2b2d42")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

        # Button to start the stream, with styling for appearance
        self.start_button = tk.Button(control_frame, text="Start Stream", font=("Arial", 12), 
                                      command=self.start_stream, bg="#4CAF50", fg="white", width=12)
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        # Button to stop the stream, initially disabled
        self.stop_button = tk.Button(control_frame, text="Stop Stream", font=("Arial", 12), 
                                     command=self.stop_stream, bg="#F44336", fg="white", width=12, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=10, pady=10)

        # Label and scale to adjust the detection sensitivity of the anomaly detector
        self.sensitivity_label = tk.Label(control_frame, text="Detection Sensitivity", 
                                          font=("Arial", 12), bg="#2b2d42", fg="#edf2f4")
        self.sensitivity_label.grid(row=0, column=2, padx=10)

        # Sensitivity scale to adjust the contamination parameter of the Isolation Forest
        self.sensitivity_scale = tk.Scale(control_frame, from_=0.01, to=0.5, resolution=0.01, 
                                          orient="horizontal", command=self.update_sensitivity, 
                                          bg="#2b2d42", fg="#edf2f4", highlightbackground="#2b2d42")
        self.sensitivity_scale.set(0.05)  # Set default sensitivity
        self.sensitivity_scale.grid(row=0, column=3, padx=10)

        # Status label to show whether the stream is idle, running, or stopped
        self.status_label = tk.Label(self.root, text="Status: Idle", font=("Arial", 14), bg="#2b2d42", fg="#edf2f4")
        self.status_label.pack(side=tk.BOTTOM, pady=10)

        # Handle window closing to ensure the stream is stopped properly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_plot(self):
        """
        Efficiently updates the plot with new data points and anomalies.

        This method optimizes the plotting process by using set_data for the line plot
        and updating scatter points for anomalies, reducing the overhead of redrawing
        the entire plot each time new data is received.
        """
        # Update the line data for the data stream
        self.line.set_data(range(len(self.data_points)), self.data_points)
        
        # Identify the x-coordinates for anomalies based on the anomaly flags
        anomalies_x = [i for i, flag in enumerate(self.anomaly_flags) if flag]
        # Extract the corresponding y-coordinates for the anomalies
        anomalies_y = [self.data_points[i] for i in anomalies_x]
        
        # Update the scatter plot for anomalies with new offsets
        self.anomaly_scatter.set_offsets(np.column_stack((anomalies_x, anomalies_y)))

        # Recalculate the limits of the axes based on the updated data
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Redraw the canvas with the updated data
        self.canvas.draw()

    def data_stream(self):
        """
        Simulates the data stream and updates the plot in real time.
        This method runs in a separate thread to avoid blocking the GUI.
        """
        self.status_label.config(text="Status: Streaming...", fg="#00bfae")  # Update status to show streaming
        while self.running:
            try:
                new_value = next(self.data_generator)  # Get the next data point from the stream
                is_anomaly = self.detector.detect(new_value)  # Check if it's an anomaly
                self.data_points.append(new_value)  # Append the data point to the list
                self.anomaly_flags.append(is_anomaly)  # Append whether it's an anomaly
                
                # Keep only the last 200 data points to maintain a sliding window effect
                if len(self.data_points) > 200:
                    self.data_points.pop(0)  # Remove the oldest data point
                    self.anomaly_flags.pop(0)  # Remove the corresponding anomaly flag
                
                self.update_plot()  # Update the plot with the new data
                time.sleep(0.05)  # Delay to simulate real-time streaming
            except StopIteration:
                break  # End the stream if the generator is exhausted
            except Exception as e:
                print(f"Stream error: {e}")
                self.status_label.config(text="Status: Error", fg="#ff3b30")
        self.status_label.config(text="Status: Idle", fg="#edf2f4")  # Set status to idle once the stream stops

    def update_sensitivity(self, val):
        """
        Updates the anomaly detection sensitivity based on the user's input from the scale.
        
        Args:
        - val: The new sensitivity value, which adjusts the contamination parameter in the Isolation Forest.
        """
        new_sensitivity = float(val)
        self.detector.model.set_params(contamination=new_sensitivity)  # Adjust the contamination parameter
        self.detector.retrain_model()  # Retrain the model with the new sensitivity

    def start_stream(self):
        """
        Starts the data stream in a separate thread, allowing real-time data to be processed and displayed.
        Disables the start button and enables the stop button while streaming.
        """
        self.running = True  # Set the running flag to True
        self.start_button.config(state=tk.DISABLED)  # Disable the start button to prevent multiple streams
        self.stop_button.config(state=tk.NORMAL)  # Enable the stop button
        threading.Thread(target=self.data_stream, daemon=True).start()  # Start the data stream in a new thread

    def stop_stream(self):
        """
        Stops the data stream by setting the running flag to False.
        Enables the start button and disables the stop button.
        """
        self.running = False  # Set the running flag to False to stop the data stream
        self.start_button.config(state=tk.NORMAL)  # Enable the start button
        self.stop_button.config(state=tk.DISABLED)  # Disable the stop button
        self.status_label.config(text="Status: Stopped", fg="#ff3b30")  # Update status to show stopped

    def on_closing(self):
        """
        Handles the window close event to ensure the program exits cleanly.
        Stops the data stream if it's running and closes the GUI.
        """
        self.running = False  # Stop the stream safely
        self.root.quit()  # Quit the Tkinter main loop
        self.root.destroy()  # Destroy the window to close the application
