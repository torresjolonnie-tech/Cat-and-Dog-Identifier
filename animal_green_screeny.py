import cv2
import numpy as np
import PIL.Image
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import time
from nanoowl.owl_predictor import OwlPredictor

# --- CONFIGURATION ---
ENGINE_PATH = "/opt/nanoowl/data/owl_image_encoder_patch32.engine"
PROMPTS = ["a cat", "a dog"]
THRESHOLD = 0.25      # Confidence threshold
CAMERA_INDEX = 0      # 0 for default USB/CSI camera
PORT = 7860           # Web port to stream to

predictor = None
text_encodings = None
cap = None

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # Convert BGR (OpenCV) to RGB PIL Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = PIL.Image.fromarray(rgb_frame)

                # Predict objects
                output = predictor.predict(
                    image=pil_image,
                    text=PROMPTS,
                    text_encodings=text_encodings,
                    threshold=THRESHOLD
                )

                detected_label = None
                if len(output.labels) > 0:
                    label_idx = output.labels[0]
                    detected_text = PROMPTS[label_idx]

                    if "cat" in detected_text:
                        detected_label = "CAT"
                    elif "dog" in detected_text:
                        detected_label = "DOG"

                # Render display output
                if detected_label:
                    h, w, _ = frame.shape
                    display_frame = np.zeros((h, w, 3), dtype=np.uint8)
                    display_frame[:] = (0, 255, 0)  # Green background

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2.0
                    thickness = 4
                    text_size = cv2.getTextSize(detected_label, font, font_scale, thickness)[0]

                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2

                    cv2.putText(
                        display_frame,
                        detected_label,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                        cv2.LINE_AA
                    )
                else:
                    display_frame = frame

                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', display_frame)
                frame_bytes = buffer.tobytes()

                try:
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                except Exception:
                    break
        else:
            self.send_error(404)

def main():
    global predictor, text_encodings, cap
    print("[DEBUG] Initializing NanoOWL predictor...", flush=True)
    predictor = OwlPredictor(
        model_name="google/owlvit-base-patch32",
        image_encoder_engine=ENGINE_PATH
    )

    print("[DEBUG] Encoding text prompts...", flush=True)
    text_encodings = predictor.encode_text(PROMPTS)

    print(f"[DEBUG] Opening camera stream at index {CAMERA_INDEX}...", flush=True)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    server = ThreadedHTTPServer(('0.0.0.0', PORT), StreamHandler)
    print(f"[SUCCESS] Streaming live on http://<JETSON-IP>:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        server.server_close()

if __name__ == '__main__':
    main()
