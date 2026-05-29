from flask import Flask, render_template, Response, request, jsonify
from detector import HandDetector
from collections import deque, Counter
import cv2
import csv
import os
import joblib
import math
from datetime import datetime
from models import db, User, History

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/sign_language_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

detector = HandDetector()
camera = cv2.VideoCapture(0)

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
CONFIRM_FRAMES = 15        
AUTO_SAVE_FRAMES = 30    

# ==========================================
# 1. HÀM MẮT THẦN (XỬ LÝ ÁNH SÁNG CLAHE)
# ==========================================
def enhance_image_for_mediapipe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    final_img = cv2.bilateralFilter(enhanced_img, 9, 75, 75)
    return final_img

# ==========================================
# 2. HÀM TIỀN XỬ LÝ TỌA ĐỘ (46 ĐẶC TRƯNG)
# ==========================================
def pre_process_landmarks(landmark_list):
    if not landmark_list: return None
    
    base_x, base_y = landmark_list[0], landmark_list[1]
    temp_list = []
    for i in range(0, len(landmark_list), 3):
        temp_list.append(landmark_list[i] - base_x)
        temp_list.append(landmark_list[i+1] - base_y)

    final_list = temp_list

    def get_dist(p1_idx, p2_idx):
        x1, y1 = final_list[p1_idx*2], final_list[p1_idx*2+1]
        x2, y2 = final_list[p2_idx*2], final_list[p2_idx*2+1]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def get_angle(p1_idx, p2_idx):
        x1, y1 = final_list[p1_idx*2], final_list[p1_idx*2+1]
        x2, y2 = final_list[p2_idx*2], final_list[p2_idx*2+1]
        return math.atan2(y2 - y1, x2 - x1)

    dists = [
        get_dist(4, 8),   # Ngón cái - Ngón trỏ
        get_dist(4, 12),  # Ngón cái - Ngón giữa
        get_dist(4, 20),  # Ngón cái - Ngón út
        get_dist(8, 20)   # Ngón trỏ - Ngón út
    ]
    
    angles = [
        get_angle(0, 4),  # Góc ngón cái
        get_angle(0, 8),  # Góc ngón trỏ
        get_angle(0, 12), # Góc ngón giữa
        get_angle(0, 16), # Góc ngón áp út
        get_angle(0, 20)  # Góc ngón út
    ]
    
    final_features = final_list + dists + angles

    max_val = max(map(abs, final_features))
    if max_val > 0:
        final_features = [n / max_val for n in final_features]
        
    return final_features

def gen_frames():
    global last_landmarks, current_result, full_sentence
    global frames_held, no_hand_frames, last_detected_char
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        # --- BẬT MẮT THẦN TRƯỚC KHI ĐƯA CHO MEDIAPIPE ---
        frame = enhance_image_for_mediapipe(frame)
        
        frame, landmarks = detector.find_hand_landmarks(frame)
        last_landmarks = landmarks 

        if landmarks and model:
            no_hand_frames = 0
            
            processed_data = pre_process_landmarks(landmarks[:63])
            try:
                probabilities = model.predict_proba([processed_data])[0]
                max_prob = max(probabilities) * 100 # Chuyển thành phần trăm
                detected = model.classes_[probabilities.argmax()] # Lấy nhãn có % cao nhất
                
                current_result = f"{detected} ({max_prob:.1f}%)"

                if max_prob >= 65.0:
                    prediction_window.append(str(detected))
                    
                    if len(prediction_window) == prediction_window.maxlen:
                        most_common = Counter(prediction_window).most_common(1)
                        stable_detected = most_common[0][0]

                        if stable_detected == last_detected_char:
                            frames_held += 1
                            if frames_held == CONFIRM_FRAMES:
                                if stable_detected == 'space':
                                    full_sentence += " "
                                elif stable_detected == 'del':
                                    full_sentence = full_sentence[:-1]
                                elif stable_detected not in ['nothing', 'No Hand']:
                                    full_sentence += stable_detected
                                print(f"📝 Cập nhật câu: {full_sentence}")
                        else:
                            last_detected_char = stable_detected
                            frames_held = 1
                else:
                    frames_held = 0
                    last_detected_char = None

            except Exception as e:
                current_result = "Lỗi nhận diện"
                print(f"Lỗi dự đoán: {e}")
                
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