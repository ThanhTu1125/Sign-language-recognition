import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import math
import seaborn as sns
import matplotlib.pyplot as plt

def train():
    csv_path = os.path.join('..', 'data', 'cleaned_data.csv')
    if not os.path.exists(csv_path):
        print("❌ LỖI: Không tìm thấy file cleaned_data.csv!")
        return

    print("📖 Đang đọc dữ liệu từ CSV...")
    data = pd.read_csv(csv_path, header=None)
    
    X = data.iloc[:, 1:].values 
    y = data.iloc[:, 0].values

    print(f"📊 Tổng số mẫu: {len(data)}")
    print(f"📐 Số lượng đặc trưng (Features): {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Đang tìm kiếm siêu tham số tối ưu (Grid Search)... Quá trình này có thể mất vài phút.")
    
    param_grid = {
        'n_estimators': [50, 100, 200],         # Thử số lượng cây khác nhau
        'max_depth': [None, 10, 20, 30],        # Thử độ sâu của cây
        'min_samples_split': [2, 5, 10]         # Thử số mẫu tối thiểu để chia nhánh
    }
    
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_ # Lấy mô hình tốt nhất
    print(f"🌟 Tham số TỐT NHẤT tìm được: {grid_search.best_params_}")

    # Đánh giá mô hình
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"✅ ĐỘ CHÍNH XÁC CỦA MÔ HÌNH: {accuracy * 100:.2f}%")
    print("-" * 30)

    # THÊM ĐOẠN NÀY VÀO ĐỂ LƯU MODEL
    models_dir = os.path.join('..', 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_path = os.path.join(models_dir, 'sign_language_model.pkl')
    joblib.dump(model, model_path)
    print(f"💾 Đã lưu mô hình thành công tại: {model_path}")

if __name__ == "__main__":
    train()