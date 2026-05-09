import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import math
import seaborn as sns
import matplotlib.pyplot as plt

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

    dists = [
        get_dist(4, 8),   # Ngón cái - Ngón trỏ
        get_dist(4, 12),  # Ngón cái - Ngón giữa
        get_dist(4, 20),  # Ngón cái - Ngón út
        get_dist(8, 20)   # Ngón trỏ - Ngón út
    ]
    
    final_features = final_list + dists

    max_val = max(map(abs, final_features))
    if max_val > 0:
        final_features = [n / max_val for n in final_features]
        
    return final_features

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
    
    print("📊 Đang vẽ ma trận nhầm lẫn (Confusion Matrix)...")
    cm = confusion_matrix(y_test, y_pred)
    
    labels = [str(cls) for cls in model.classes_]
    
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    plt.figure(figsize=(14, 12))
    
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    
    plt.title("Ma trận nhầm lẫn (Confusion Matrix)")
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    
    cm_path = os.path.join(models_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"🖼️ Đã lưu ảnh ma trận nhầm lẫn tại: {cm_path}")
    
    plt.show()

if __name__ == "__main__":
    train()