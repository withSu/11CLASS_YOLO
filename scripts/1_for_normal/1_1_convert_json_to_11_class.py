# convert_json.py

import os
import json

# 클래스 이름과 ID 매핑
category_map = {
    'Chip': 0,
    'CSolder': 1,
    '2sideIC': 2,
    'SOD': 3,
    'Circle': 4,
    '4sideIC': 5,
    'Tantalum': 6,
    'BGA': 7,
    'MELF': 8,
    'Crystal': 9,
    'Array': 10
}

def process_json_files(input_folder, output_folder, category_map):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            new_shapes = []
            for shape in data.get("shapes", []):
                label = shape["label"]
                
                # category_map에 존재하는 클래스만 남기기
                if label in category_map:
                    # label을 그대로 두고, category_id를 매핑된 값으로 저장
                    shape["category_id"] = category_map[label]
                    new_shapes.append(shape)
            
            data["shapes"] = new_shapes

            # 변경된 JSON을 출력 폴더에 저장
            output_file_path = os.path.join(output_folder, filename)
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                json.dump(data, output_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    input_folder = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/2_raw_json_labels"
    output_folder = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/3_new_raw_json_labels"
    process_json_files(input_folder, output_folder, category_map)
