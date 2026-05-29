import pandas as pd
import os
from sklearn.ensemble import IsolationForest

def clean_data():
    input_path = os.path.join('..', 'data', 'hand_data_all.csv')
    output_path = os.path.join('..', 'data', 'cleaned_data.csv')
    
    if not os.path.exists(input_path):
        print(f"❌ Không thấy file {input_path}")
        return

    print("🧹 Đang lọc dữ liệu nhiễu (Chế độ Per-Class)...")
    df = pd.read_csv(input_path, header=None)
    
    cleaned_chunks = []
    
    labels = df[0].unique()
    
    for label in labels:
        group = df[df[0] == label]
        coords = group.iloc[:, 1:]
        
        iso = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso.fit_predict(coords)
        
        good_data = group[outliers == 1]
        cleaned_chunks.append(good_data)
        
        print(f"  👉 Đã dọn nhãn [{label}]: Giữ lại {len(good_data)}/{len(group)} dòng.")
        
    df_clean = pd.concat(cleaned_chunks, ignore_index=True)
    
    df_clean.to_csv(output_path, index=False, header=None)
    print("="*50)
    print(f"✅ Hoàn tất dọn dẹp tổng thể! Giữ lại {len(df_clean)}/{len(df)} dòng.")
    print(f"📍 File sạch lưu tại: {output_path}")

if __name__ == "__main__":
    clean_data()