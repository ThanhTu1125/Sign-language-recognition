import cv2
import os
import csv
import warnings
import math
from tqdm import tqdm
try:
    import mediapipe as mp
    print(f"✅ MediaPipe đang chạy từ: {mp.__file__}")
except AttributeError:
    print("❌ LỖI: Thư viện MediaPipe bị nhận diện sai. Kiểm tra xem có file mediapipe.py nào trong thư mục không!")

from detector import HandDetector

warnings.filterwarnings("ignore", category=UserWarning)

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

def batch_extract():
    detector = HandDetector()
    # train_dir = os.path.join('..', 'data', 'asl_alphabet_train_tus')
    # output_file = os.path.join('..', 'data', 'hand_data.csv')
    train_dir = os.path.join('..', 'data', 'asl_alphabet_train_J_N_M_O_U_V_Z')
    output_file = os.path.join('..', 'data', 'hand_data_J_N_M_O_U_V_Z.csv')

    if not os.path.exists(train_dir):
        print(f"❌ Không tìm thấy thư mục dữ liệu tại: {train_dir}")
        return

    labels = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    labels.sort()

    print(f"🚀 Bắt đầu quét {len(labels)} nhãn...")

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        for label in labels:
            label_path = os.path.join(train_dir, label)
            images = [img for img in os.listdir(label_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
            
            pbar = tqdm(images, desc=f"📦 Nhãn [{label}]", unit="img")
            for img_name in pbar:
                img = cv2.imread(os.path.join(label_path, img_name))
                if img is None: continue
                
                _, landmarks = detector.find_hand_landmarks(img)
                if landmarks:
                    clean_data = pre_process_landmarks(landmarks[:63])
                    if clean_data:
                        if clean_data is not None:
                            writer.writerow([label] + list(clean_data))
            pbar.close()

    print(f"✅ HOÀN THÀNH! Dữ liệu đã lưu tại: {output_file}")

if __name__ == "__main__":
    batch_extract()