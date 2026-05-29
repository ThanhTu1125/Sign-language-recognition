from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from detector import HandDetector
from collections import deque, Counter
import cv2
import csv
import os
import joblib
import math
from datetime import datetime
from models import db, User, History, SignData

app = Flask(__name__)

# CẤU HÌNH BẢO MẬT & DATABASE
app.secret_key = 'sign_language_super_secret_key_123' 
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

prediction_window = deque(maxlen=10) # Bộ đệm lọc nhiễu 10 khung hình
full_sentence = ""         
last_landmarks = []
current_result = "No Hand"

frames_held = 0            
no_hand_frames = 0         
last_detected_char = None  
CONFIRM_FRAMES = 20       
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
# 2. HÀM TIỀN XỬ LÝ TỌA ĐỘ (51 ĐẶC TRƯNG CÓ GÓC)
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
        get_dist(4, 8),   
        get_dist(4, 12),  
        get_dist(4, 20),  
        get_dist(8, 20)   
    ]
    
    angles = [
        get_angle(0, 4),  
        get_angle(0, 8),  
        get_angle(0, 12), 
        get_angle(0, 16), 
        get_angle(0, 20)  
    ]
    
    final_features = final_list + dists + angles

    max_val = max(map(abs, final_features))
    if max_val > 0:
        final_features = [n / max_val for n in final_features]
        
    return final_features

def save_to_db(text, user_id):
    with app.app_context():
        try:
            new_entry = History(user_id=user_id, result_text=text)
            db.session.add(new_entry)
            db.session.commit()
        except Exception as e:
            print(f"Lỗi lưu DB: {e}")

# ==========================================
# 3. LUỒNG CAMERA NHẬN DIỆN
# ==========================================
def gen_frames(current_user_id):
    global last_landmarks, current_result, full_sentence
    global frames_held, no_hand_frames, last_detected_char
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        frame = enhance_image_for_mediapipe(frame)
        frame, landmarks = detector.find_hand_landmarks(frame)
        last_landmarks = landmarks 

        if landmarks and model:
            no_hand_frames = 0
            processed_data = pre_process_landmarks(landmarks[:63])
            try:
                probabilities = model.predict_proba([processed_data])[0]
                max_prob = max(probabilities) * 100 
                detected = model.classes_[probabilities.argmax()] 
                
                # Vẫn hiển thị % lên màn hình để biết AI đang nghĩ gì
                current_result = f"{detected} ({max_prob:.1f}%)"

                # ĐÃ LOẠI BỎ IF max_prob >= 65.0
                # Chấp nhận mọi kết quả cao nhất, bộ lọc sẽ dùng thời gian giữ tay để quyết định
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
                            # Reset đếm để tránh bị gõ liên tiếp 1 chữ quá nhanh
                            frames_held = 0 
                    else:
                        last_detected_char = stable_detected
                        frames_held = 1

            except Exception as e:
                current_result = "Lỗi nhận diện"
                
        elif not landmarks:
            prediction_window.clear()
            current_result = "No Hand"
            last_detected_char = None
            frames_held = 0
            
            if full_sentence.strip(): 
                no_hand_frames += 1
                if no_hand_frames >= AUTO_SAVE_FRAMES:
                    save_to_db(full_sentence.strip(), current_user_id)
                    print(f"💾 Tự động lưu và kết thúc câu: {full_sentence}")
                    full_sentence = ""     
                    no_hand_frames = 0     

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==========================================
# 4. HỆ THỐNG ĐĂNG KÝ / ĐĂNG NHẬP
# ==========================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            return render_template('register.html', error="Vui lòng điền đầy đủ thông tin!")
        
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            return render_template('register.html', error="Tên đăng nhập hoặc Email đã tồn tại!")
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password, role='user')
        
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            return render_template('login.html', error="Vui lòng nhập tài khoản và mật khẩu!")
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            return redirect(url_for('index'))
            
        return render_template('login.html', error="Sai tài khoản hoặc mật khẩu!")
        
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ==========================================
# 5. CÁC API & ROUTE GIAO DIỆN CHÍNH
# ==========================================
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    current_user_id = session.get('user_id', 1)
    return Response(gen_frames(current_user_id), mimetype='multipart/x-mixed-replace; boundary=frame')

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
        if full_sentence.strip() and 'user_id' in session:
            save_to_db(full_sentence, session['user_id'])
            full_sentence = ""
    return jsonify({"status": "success"})

@app.route('/save_history', methods=['POST'])
def save_history():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Chưa đăng nhập"})
    data = request.json
    try:
        new_entry = History(user_id=session['user_id'], result_text=data['text'])
        db.session.add(new_entry)
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_history_list')
def get_history_list():
    if 'user_id' not in session:
        return jsonify([])
    
    if session.get('role') == 'admin':
        histories = History.query.order_by(History.created_at.desc()).limit(10).all()
    else:
        histories = History.query.filter_by(user_id=session['user_id']).order_by(History.created_at.desc()).limit(10).all()
        
    output = []
    for h in histories:
        output.append({
            "id": h.id, "result": h.result_text, "time": h.created_at.strftime("%H:%M:%S %d/%m/%Y")
        })
    return jsonify(output)

@app.route('/collect_data', methods=['POST'])
def collect_data():
    if session.get('role') != 'admin':
        return jsonify({"status": "error", "message": "Không đủ quyền"})
        
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

@app.route('/api/dictionary', methods=['GET'])
def get_dictionary():
    signs = SignData.query.all()
    result = []
    for s in signs:
        result.append({
            'id': s.id,
            'sign_name': s.sign_name,
            'description': s.description,
            'image_url': f"/static/images/signs/{s.sample_image_path}" if s.sample_image_path else ""
        })
    return jsonify(result)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # 1. KIỂM TRA TÀI KHOẢN ADMIN
        admin = User.query.filter_by(username='admin').first()
        
        if not admin:
            hashed_pw = generate_password_hash('123')
            admin = User(username='admin', email='admin@gmail.com', password_hash=hashed_pw, role='admin')
            db.session.add(admin)
            db.session.commit()
            print("✅ Đã tạo User Admin mặc định (Tài khoản: admin / Pass: 123)")
            
        elif admin.password_hash == '123':
            admin.password_hash = generate_password_hash('123')
            db.session.commit()
            print("✅ Đã tự động băm lại mật khẩu cho tài khoản Admin cũ!")

        # 2. TỰ ĐỘNG NẠP 28 DỮ LIỆU TỪ ĐIỂN
        if not SignData.query.first():
            vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del']
            admin_id = admin.id if admin else 1
            
            for label in vocab:
                new_sign = SignData(
                    sign_name=label,
                    description=f"Đây là ký hiệu cho: {label}",
                    sample_image_path=f"{label} (1).jpg", 
                    created_by=admin_id
                )
                db.session.add(new_sign)
            db.session.commit()
            print("✅ Đã tự động tạo 28 dữ liệu mẫu cho Từ điển Ký hiệu!")

        print("✅ Database đã sẵn sàng!")
    app.run(debug=True)