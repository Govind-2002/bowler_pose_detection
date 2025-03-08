from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import time
import os
from main import BowlingAnalyzer  # Import your bowling analysis class

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Global variable to store analysis results
analysis_results = {}

# Initialize camera for webcam feed
class Camera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)  # Initialize camera
        time.sleep(2)  # Warm-up time for camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.analyzer = BowlingAnalyzer()  # Initialize your bowling analyzer

    def get_frame(self):
        success, frame = self.camera.read()
        if not success:
            return None
        # Flip frame horizontally for a more intuitive view
        frame = cv2.flip(frame, 1)
        # Run your bowling analysis on the frame
        analyzed_frame = self.analyzer.analyze_frame(frame)
        # Convert to JPEG format
        ret, jpeg = cv2.imencode('.jpg', analyzed_frame)
        return jpeg.tobytes()

    def release(self):
        self.camera.release()

# Video upload and processing
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_video_frames(filepath, filename):
    analyzer = BowlingAnalyzer()
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file {filepath}")
        return
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video playback finished")
            # Store analysis results globally
            analysis_results[filename] = analyzer.get_final_result()
            break
        
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Run your bowling analysis on the frame
        analyzed_frame = analyzer.analyze_frame(frame)
        
        # Convert to JPEG format
        ret, jpeg = cv2.imencode('.jpg', analyzed_frame)
        if not ret:
            print("ERROR: Failed to encode frame")
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    
    cap.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        camera = Camera()
        try:
            while True:
                frame = camera.get_frame()
                if frame is None:
                    break
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            camera.release()
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        print(f"Video saved to: {filename}")
        
        # Redirect to the video analysis page
        return redirect(url_for('analyze_video', filename=file.filename))
    
    return redirect(request.url)

@app.route('/analyze/<filename>')
def analyze_video(filename):
    return render_template('analyze_video.html', filename=filename)

@app.route('/video_analysis_feed/<filename>')
def video_analysis_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Streaming video from: {filepath}")
    return Response(generate_video_frames(filepath, filename),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results/<filename>')
def show_results(filename):
    if filename in analysis_results:
        return render_template('results.html', result=analysis_results[filename])
    return redirect(url_for('index'))

@app.route('/check_video_end/<filename>')
def check_video_end(filename):
    # Check if analysis results are available for this video
    if filename in analysis_results:
        return jsonify({"finished": True})
    return jsonify({"finished": False})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)