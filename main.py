import tkinter as tk
from visualization import RealTimeGUI

def main():
    """
    The main entry point for the real-time anomaly detection application.
    Initializes the Tkinter root window and launches the GUI.
    """
    root = tk.Tk()
    app = RealTimeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
