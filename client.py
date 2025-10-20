import requests
import numpy as np
import io

def fetch_numpy_array(url="http://192.168.0.228:8000/capture_array"):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Assuming the response contains a serialized NumPy array (e.g., in .npy format)
        array_data = io.BytesIO(response.content)
        array = np.load(array_data)  # Deserialize the NumPy array

        return array

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except Exception as e:
        print(f"Error processing the array: {e}")
        return None
