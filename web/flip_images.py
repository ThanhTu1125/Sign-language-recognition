import cv2
import os
from tqdm import tqdm

def augment_left_hand():
    # Trỏ vào thư mục chứa dữ liệu gốc
    dataset_dir = os.path.join('..', 'data', 'asl_alphabet_train_J_N_M_O_U_V_Z')
    
    if not os.path.exists(dataset_dir):
        print("❌ Không tìm thấy thư mục dữ liệu!")
        return

    labels = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    print("🔄 BẮT ĐẦU NHÂN BẢN DỮ LIỆU TAY TRÁI...")

    for label in labels:
        label_path = os.path.join(dataset_dir, label)
        images = [img for img in os.listdir(label_path) if img.endswith(('.jpg', '.png'))]
        
        # Chỉ lật ảnh gốc, bỏ qua ảnh đã lật trước đó để tránh lặp vô tận
        original_images = [img for img in images if not img.startswith('left_')]
        
        pbar = tqdm(original_images, desc=f"Lật nhãn [{label}]", unit="img")
        for img_name in pbar:
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                # cv2.flip(img, 1) để lật theo trục dọc (như soi gương)
                flipped_img = cv2.flip(img, 1)
                
                # Lưu file mới với tiền tố 'left_'
                new_img_name = f"left_{img_name}"
                new_img_path = os.path.join(label_path, new_img_name)
                cv2.imwrite(new_img_path, flipped_img)
                
        pbar.close()
        
    print("✅ HOÀN THÀNH! Số lượng ảnh đã được nhân đôi.")

if __name__ == "__main__":
    augment_left_hand()