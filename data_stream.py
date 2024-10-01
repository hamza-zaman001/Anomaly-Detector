import random
import time

def data_stream_generator(num_points=1000, anomaly_ratio=0.05):
    """
    Simulates a real-time data stream with regular patterns, random noise, and occasional anomalies.
    
    Args:
    - num_points: The total number of data points to generate (default is 1000).
    - anomaly_ratio: The proportion of data points that will be anomalies (default is 5%).
    
    Yields:
    - A single float value from the simulated data stream.
    """
    data = []
    
    for i in range(num_points):
        try:
            base_value = 100 + random.uniform(-10, 10)
            seasonal_effect = i % 100
            value = base_value + seasonal_effect
            if random.random() < anomaly_ratio:
                value += random.uniform(50, 100)
            if random.random() < 0.01:  # Introduce some random data failures
                value = None  # Simulating a corrupt data point
            if value is not None:
                yield value
        except Exception as e:
            print(f"Data stream error: {e}")
        time.sleep(0.01)
    
    # Stream the data one point at a time
    for value in data:
        yield value
        time.sleep(0.01)  # Simulate real-time streaming by adding a small delay
