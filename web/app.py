from flask import Flask, render_template, Response, request, jsonify
from detector import HandDetector
from collections import deque, Counter
import cv2
import csv
import os
import joblib
import math
from datetime import datetime

try:
    from models import db, User, History
except ImportError:
    from .models import db, User, History

app = Flask(__name__)

# Cấu hình Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/sign_language_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

detector = HandDetector()
camera = cv2.VideoCapture(0)

# Load Model
MODEL_PATH = os.path.join('..', 'models', 'sign_language_model.pkl')
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Đã nạp mô hình AI thành công!")
else:
    print("⚠️ CẢNH BÁO: Không tìm thấy file mô hình!")

prediction_window = deque(maxlen=10)
full_sentence = ""         # Câu hoàn chỉnh
last_landmarks = []
current_result = "No Hand"

frames_held = 0            
no_hand_frames = 0         
last_detected_char = None  
CONFIRM_FRAMES = 25        
AUTO_SAVE_FRAMES = 60    

def pre_process_landmarks(landmark_list):
    if not landmark_list: return None
    base_x, base_y = landmark_list[0], landmark_list[1]
    temp_list = []
    for i in range(0, len(landmark_list), 3):
        temp_list.append(landmark_list[i] - base_x)
        temp_list.append(landmark_list[i+1] - base_y)
    val_x, val_y = temp_list[18], temp_list[19]
    angle = math.atan2(val_x, -val_y) 
    final_list = []
    c, s = math.cos(-angle), math.sin(-angle)
    for i in range(0, len(temp_list), 2):
        x, y = temp_list[i], temp_list[i+1]
        final_list.append(x * c - y * s)
        final_list.append(x * s + y * c)
    max_val = max(map(abs, final_list))
    if max_val > 0:
        final_list = [n / max_val for n in final_list]
    return final_list

def gen_frames():
    global last_landmarks, current_result, full_sentence
    global frames_held, no_hand_frames, last_detected_char
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        frame, landmarks = detector.find_hand_landmarks(frame)
        last_landmarks = landmarks 

        if landmarks and model:
            no_hand_frames = 0
            
            processed_data = pre_process_landmarks(landmarks[:63])
            try:
                raw_prediction = model.predict([processed_data])[0]
                prediction_window.append(str(raw_prediction))
                most_common = Counter(prediction_window).most_common(1)
                detected = most_common[0][0]
                
                current_result = detected

                if detected == last_detected_char:
                    frames_held += 1
                    if frames_held == CONFIRM_FRAMES:
                        if detected == 'space':
                            full_sentence += " "
                        elif detected == 'del':
                            full_sentence = full_sentence[:-1]
                        elif detected not in ['nothing', 'No Hand']:
                            full_sentence += detected
                        print(f"📝 Cập nhật câu: {full_sentence}")
                else:
                    last_detected_char = detected
                    frames_held = 1

            except Exception:
                current_result = "Lỗi nhận diện"
                
        elif not landmarks:
            prediction_window.clear()
            current_result = "No Hand"
            last_detected_char = None
            frames_held = 0
            
            if full_sentence.strip(): 
                no_hand_frames += 1
                if no_hand_frames >= AUTO_SAVE_FRAMES:
                    save_to_db(full_sentence.strip())
                    print(f"💾 Tự động lưu và kết thúc câu: {full_sentence}")
                    full_sentence = ""     # Xóa màn hình
                    no_hand_frames = 0     # Reset

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def save_to_db(text):
    with app.app_context():
        try:
            new_entry = History(user_id=1, result_text=text)
            db.session.add(new_entry)
            db.session.commit()
        except Exception as e:
            print(f"Lỗi lưu DB: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    return jsonify({
        "current_char": last_detected_char if last_detected_char else "--",
        "full_sentence": full_sentence,
        "raw_result": current_result
    })

@app.route('/control', methods=['POST'])
def control():
    # Đã xóa current_char_buffer khỏi khai báo global
    global full_sentence
    action = request.json.get('action')
    if action == 'clear':
        full_sentence = ""
    elif action == 'backspace':
        full_sentence = full_sentence[:-1]
    elif action == 'space':
        full_sentence += " "
    elif action == 'save':
        if full_sentence.strip():
            save_to_db(full_sentence)
            full_sentence = ""
    return jsonify({"status": "success"})

@app.route('/save_history', methods=['POST'])
def save_history():
    data = request.json
    try:
        new_entry = History(user_id=1, result_text=data['text'])
        db.session.add(new_entry)
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_history_list')
def get_history_list():
    histories = History.query.order_by(History.created_at.desc()).limit(10).all()
    output = []
    for h in histories:
        output.append({
            "id": h.id, "result": h.result_text, "time": h.created_at.strftime("%H:%M:%S %d/%m/%Y")
        })
    return jsonify(output)

@app.route('/collect_data', methods=['POST'])
def collect_data():
    global last_landmarks
    data = request.json
    label = data.get('label')
    if not last_landmarks:
        return jsonify({"status": "error", "message": "No landmarks detected"})
    
    clean_data = pre_process_landmarks(last_landmarks[:63])

    if not os.path.exists('../data'):
        os.makedirs('../data')
    
    file_path = '../data/hand_data_web.csv'    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if clean_data:
            writer.writerow([label] + list(clean_data))
            
    return jsonify({"status": "success"})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.get(1):
            admin = User(id=1, username='admin', email='admin@gmail.com', password_hash='123')
            db.session.add(admin)
            db.session.commit()
            print("✅ Đã tạo User mặc định (ID: 1)")
        print("✅ Database đã sẵn sàng!")
    app.run(debug=True)