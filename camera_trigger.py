import time
import threading
import cv2
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

# --- Setup ---
app = Flask(__name__)
picam2 = Picamera2()

# Configure for exactly one stream at full sensor resolution
config = picam2.create_still_configuration(main={"size": (1456, 1088)})
picam2.configure(config)
picam2.start()

# HTML for the web page
HTML = """
<html>
  <head><title>Global Shutter Full Res</title></head>
  <body style="background: #222; color: white; text-align: center; font-family: sans-serif;">
    <h1>📸 Global Shutter (1456x1088)</h1>
    <img src="/video_feed" style="width: 80%; border: 5px solid #444; border-radius: 10px;">
    <p>Status: <b>Live Stream Running</b></p>
    <p>Press <b>ENTER</b> in your Terminal to save 1456x1088 JPGs.</p>
  </body>
</html>
"""

def generate_frames():
    while True:
        # Capture the full-res frame (since it's the only stream available)
        frame = picam2.capture_array() 
        
        # Resize for the web feed so the browser stays responsive
        # This doesn't affect the saved file quality
        preview_frame = cv2.resize(frame, (728, 544)) 
        preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_RGB2BGR)
        
        ret, buffer = cv2.imencode('.jpg', preview_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def terminal_logic():
    count = 1
    time.sleep(2) 
    print("\n" + "="*40)
    print("SENSOR RESOLUTION LOCKED: 1456 x 1088")
    print("="*40 + "\n")
    
    try:
        while True:
            input(f"Ready for calib{count}.jpg? (Press Enter) ")
            for i in range(3, 0, -1):
                print(f"📸 Snapping in {i}...", end="\r")
                time.sleep(1)
            
            filename = f"calib{count}.jpg"
            # Captures the current configuration (1456x1088) to file
            picam2.capture_file(filename)
            
            print(f"\n✅ Saved: {filename} (1456x1088)           ")
            count += 1
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    # Start terminal logic thread
    threading.Thread(target=terminal_logic, daemon=True).start()
    # Start Flask (host '0.0.0.0' allows other devices on network to see the feed)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
