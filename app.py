import argparse
import os
import cv2
import time
import requests
from flask import Flask, render_template, request, send_from_directory, Response, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import shutil

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

def send_image_to_spring_boot(image_path):
    spring_boot_url = 'http://192.168.115.98:9090/processImage'  # Replace <SPRING_BOOT_IP> and <PORT> with your Spring Boot's IP and port
    with open(image_path, 'rb') as image_file:
        files = {'file': image_file}
        response = requests.post(spring_boot_url, files=files)
    return response

@app.route("/receiveImage", methods=["POST"])
def receive_image():
    data = request.json
    image_url = data.get('imageUrl')
    print(image_url)
    if image_url:
        response = requests.get(image_url)
        if response.status_code == 200:
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', secure_filename(image_url.split("/")[-1]))
            with open(filepath, 'wb') as f:
                f.write(response.content)
            # Proceed with image processing
            model = YOLO('best.pt')
            img = cv2.imread(filepath)
            detections = model(img, save=True)
            
            # Send processed image to Spring Boot
            spring_boot_response = send_image_to_spring_boot(filepath)
            if spring_boot_response.status_code == 200:
                return jsonify({"message": "Image processed and sent successfully", "image_path": filepath}), 200
            else:
                return jsonify({"message": "Failed to send image to Spring Boot"}), 500
        else:
            return jsonify({"message": "Failed to download image"}), 400
    return jsonify({"message": "No image URL provided"}), 400


"""@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            print("Upload folder is ", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print("Printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            model = YOLO('best.pt')

            if file_extension == 'jpg':
                img = cv2.imread(filepath)

                # Perform the detection
                detections = model(img, save=True)
                return display(f.filename)

            elif file_extension == 'mp4':
                video_path = filepath  # replace with your video path
                cap = cv2.VideoCapture(video_path)

                # Get video dimensions
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, save=True)
                    print(results)
                    cv2.waitKey(1)

                    res_plotted = results[0].plot()
                    cv2.imshow("result", res_plotted)

                    # Write the frame to the output video
                    out.write(res_plotted)

                return video_feed()

        return render_template('index.html', error="No file uploaded or unsupported file format.")
    
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    if subfolders:
        latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
        image_path = os.path.join(folder_path, latest_subfolder, f.filename)
        return render_template('index.html', image_path=image_path)
    
    return render_template('index.html', error="No detections found.")

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(filepath)

            model = YOLO('best.pt')

            # Check file extension
            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension == 'jpg':
                img = cv2.imread(filepath)

                # Perform YOLOv8 detection
                detections = model(img, save=True)

                # YOLOv8 saves the result in the 'runs' directory by default
                folder_path = 'runs/detect'
                subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
                latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
                image_path = os.path.join(folder_path, latest_subfolder, f.filename)

                # Copy or move the detected image to the static folder
                static_image_path = os.path.join(basepath, 'static', f.filename)
                shutil.copy(image_path, static_image_path)

                # Now pass this path to the HTML template for rendering
                return render_template('index.html', image_path=static_image_path)

        return render_template('index.html', error="No file uploaded or unsupported file format.")"""

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            upload_folder = os.path.join(basepath, 'uploads')
            static_folder = os.path.join(basepath, 'static')
            filepath = os.path.join(upload_folder, secure_filename(f.filename))
            f.save(filepath)

            model = YOLO('best.pt')

            # Perform YOLOv8 detection and save the prediction
            detections = model(filepath, save=True)

            # Get the latest saved prediction image path
            folder_path = 'runs/detect'
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
            image_filename = f.filename
            image_path = os.path.join(folder_path, latest_subfolder, image_filename)

            # Debugging statements
            print(f"Image Path: {image_path}")
            print(f"Static Image Path: {os.path.join(static_folder, image_filename)}")

            # Ensure the file exists before copying
            if os.path.exists(image_path):
                shutil.copy(image_path, os.path.join(static_folder, image_filename))
                return render_template('index.html', image_path=image_filename)
            else:
                print("Image file not found.")
                return render_template('index.html', error="Image file not found.")
    
    # Render the template without an image for GET requests or if something fails
    return render_template('index.html', image_path=None)



@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, latest_subfolder)
    print("printing directory: ", directory)
    files = os.listdir(directory)
    latest_file = files[0]

    print(latest_file)

    filename = os.path.join(directory, latest_file)
    file_extension = filename.rsplit('.', 1)[1].lower()

    if file_extension == 'jpg':
        return send_from_directory(directory, latest_file)  # Remove 'environ' argument

    return "Invalid file format"

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)  # control the frame rate to display one frame every 100 milliseconds

@app.route("/video_feed")
def video_feed():
    print("function called")

    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO('best.pt')
    app.run(host="0.0.0.0",port=args.port)
