# input: 2_raw_json
# output: 3_new_raw_json
import os
import json

def process_json_files(input_folder, output_folder, classes_to_keep, new_class_name):
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
                
                # 원하는 클래스만 남기기
                if label in classes_to_keep:
                    # 모든 남은 클래스의 이름을 하나로 변경
                    shape["label"] = new_class_name
                    new_shapes.append(shape)
            
            data["shapes"] = new_shapes

            # 변경된 JSON을 출력 폴더에 저장
            output_file_path = os.path.join(output_folder, filename)
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                json.dump(data, output_file, indent=4, ensure_ascii=False)

# 사용 예제
input_folder = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset2/2_raw_json"  # JSON 파일들이 담긴 폴더 경로
output_folder = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset2/3_new_raw_json"  # 수정된 JSON 파일들을 저장할 폴더 경로
classes_to_keep = {"AL-Capacitor", "BGA", "C-chip", "Crystal", "DPAK", "Inductor",
    "L-chip", "LED", "MELF", "PLCC(Square)", "QFN(Rectangular)", "QFN(Square)",
    "QFP(Square)", "R-chip", "ResistorsChipArray", "SOD", "SOIC", "SON", "SOP",
    "SOT", "TSOP", "Tantalum", "CMounting","Chip","Array","4sideIC","2sideIC","Circle"}  # 남기고 싶은 클래스들
new_class_name = "component"

process_json_files(input_folder, output_folder, classes_to_keep, new_class_name)
