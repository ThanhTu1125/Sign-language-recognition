import pandas as pd
import os

def merge_csv_files():
    data_dir = os.path.join('..', 'data')
    output_file = os.path.join(data_dir, 'hand_data_all.csv')
    
    files_to_merge = ['hand_data.csv', 'hand_data_web_1.csv','hand_data_web.csv']
    
    combined_data = []
    
    print("📂 Đang bắt đầu gộp các file dữ liệu...")
    
    for file in files_to_merge:
        file_path = os.path.join(data_dir, file)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)
            combined_data.append(df)
            print(f"✅ Đã nạp: {file} ({len(df)} dòng)")
        else:
            print(f"⚠️ CẢNH BÁO: Không tìm thấy file {file}")

    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        
        final_df.to_csv(output_file, index=False, header=None)
        print("-" * 30)
        print(f"🚀 HOÀN THÀNH! Tổng cộng có {len(final_df)} dòng dữ liệu.")
        print(f"📍 File đã lưu tại: {output_file}")
    else:
        print("❌ Không có dữ liệu để gộp!")

if __name__ == "__main__":
    merge_csv_files()