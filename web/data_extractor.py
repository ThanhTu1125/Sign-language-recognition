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

def batch_extract():
    detector = HandDetector()
    train_dir = os.path.join('..', 'data', 'asl_alphabet_train_tus')
    output_file = os.path.join('..', 'data', 'hand_data.csv')

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