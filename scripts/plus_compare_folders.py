import os

def compare_folders(folder1, folder2):
    folder1_files = set(os.listdir(folder1))
    folder2_files = set(os.listdir(folder2))
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    json_extension = '.json'
    
    folder1_images = {f for f in folder1_files if os.path.splitext(f)[1].lower() in image_extensions}
    folder2_images = {f for f in folder2_files if os.path.splitext(f)[1].lower() in image_extensions}
    
    folder1_jsons = {f for f in folder1_files if f.endswith(json_extension)}
    folder2_jsons = {f for f in folder2_files if f.endswith(json_extension)}
    
    unmatched_images_in_folder1 = folder1_images - folder2_images
    unmatched_images_in_folder2 = folder2_images - folder1_images
    
    unmatched_jsons_in_folder1 = folder1_jsons - folder2_jsons
    unmatched_jsons_in_folder2 = folder2_jsons - folder1_jsons
    
    print("=== 매칭되지 않는 이미지 파일 ===")
    print("폴더1에만 있는 이미지 파일:", unmatched_images_in_folder1)
    print("폴더2에만 있는 이미지 파일:", unmatched_images_in_folder2)
    
    print("\n=== 매칭되지 않는 JSON 파일 ===")
    print("폴더1에만 있는 JSON 파일:", unmatched_jsons_in_folder1)
    print("폴더2에만 있는 JSON 파일:", unmatched_jsons_in_folder2)

# 🔥 폴더 경로를 여기에 직접 입력
folder1_path = "/home/a/A_2024_selfcode/PCB/dataset/0_raw_data/kbs_dragged2/before"
folder2_path = "/home/a/A_2024_selfcode/PCB/GT/kbs_only_rect_and_draged"

compare_folders(folder1_path, folder2_path)
