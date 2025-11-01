import cv2
import io
import numpy as np
import requests


CAMERA_URL = "http://192.168.0.228:8000"


def fetch_numpy_array(
    url=f"{CAMERA_URL}/capture_array",
    high_res=False,
):
    try:
        response = requests.get(
            url,
            params={"high_res": high_res},
        )
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


def fetch_jpeg(
    url=f"{CAMERA_URL}/capture_jpeg",
    high_res=False,
):
    """
    Fetch a JPEG image from the camera service and return it as an OpenCV BGR array.
    """
    try:
        response = requests.get(
            url,
            params={"high_res": high_res},
        )
        response.raise_for_status()  # HTTP errors

        # Convert raw JPEG bytes into a NumPy array, then decode into an image
        img_buf = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
        return img

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except Exception as e:
        print(f"Error decoding JPEG: {e}")
        return None


def mjpeg_stream_watcher(
    fn,
    url=f"{CAMERA_URL}/video_feed",
    imshow=False,
):
    stream = cv2.VideoCapture(url)

    while (stream.isOpened()):
        ret, img = stream.read()
        if ret: # ret == True if stream.read() was successful

            if not imshow:
                # process every frame
                img = fn(img)
            else:
                # display preview
                cv2.imshow('preview', img)

                # process frame
                if cv2.waitKey(1) == ord('p'):
                    print("Processing frame...")
                    try:
                        fn(img)
                    except:
                        break

                # exit loop
                if cv2.waitKey(1) == ord('q'):
                    print("Exiting...")
                    break

    # cleanup cv2
    stream.release()
    cv2.destroyAllWindows()


def dummy(img):
    print(f"Processed image {img.shape}")
    return img


if __name__ == "__main__":
    mjpeg_stream_watcher(
        fn=dummy,
        imshow=True,
    )
