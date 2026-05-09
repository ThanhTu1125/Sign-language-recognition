import pandas as pd
import os
from sklearn.ensemble import IsolationForest

def clean_data():
    input_path = os.path.join('..', 'data', 'hand_data_all.csv')
    output_path = os.path.join('..', 'data', 'cleaned_data.csv')
    
    if not os.path.exists(input_path):
        print(f"❌ Không thấy file {input_path}")
        return

    print("🧹 Đang lọc dữ liệu nhiễu...")
    df = pd.read_csv(input_path, header=None)
    
    labels = df.iloc[:, 0]
    coords = df.iloc[:, 1:]
    
    iso = IsolationForest(contamination=0.05, random_state=42) # Lọc bỏ 5% dữ liệu lỗi nhất
    outliers = iso.fit_predict(coords)
    
    df_clean = df[outliers == 1]
    
    df_clean.to_csv(output_path, index=False, header=None)
    print(f"✅ Đã dọn dẹp! Giữ lại {len(df_clean)}/{len(df)} dòng.")
    print(f"📍 File sạch lưu tại: {output_path}")

if __name__ == "__main__":
    clean_data()