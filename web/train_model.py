import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import math

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

def train():
    csv_path = os.path.join('..', 'data', 'cleaned_data.csv')
    if not os.path.exists(csv_path):
        print("❌ LỖI: Không tìm thấy file cleaned_data.csv! Hãy chạy data_extractor.py trước.")
        return

    print("📖 Đang đọc dữ liệu từ CSV...")
    data = pd.read_csv(csv_path, header=None)
    
    X = data.iloc[:, 1:].values 
    y = data.iloc[:, 0].values

    print(f"📊 Tổng số mẫu: {len(data)}")
    print(f"📐 Số lượng đặc trưng (Features): {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"🧠 Đang huấn luyện mô hình Random Forest với {len(X_train)} mẫu...")

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"✅ ĐỘ CHÍNH XÁC CỦA MÔ HÌNH: {accuracy * 100:.2f}%")
    print("-" * 30)

    models_dir = os.path.join('..', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_path = os.path.join(models_dir, 'sign_language_model.pkl')
    joblib.dump(model, model_path)
    print(f"💾 Đã lưu mô hình thành công tại: {model_path}")

if __name__ == "__main__":
    train()