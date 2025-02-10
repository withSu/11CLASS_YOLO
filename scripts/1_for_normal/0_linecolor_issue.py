# input: 2_raw_json
# output: 3_new_raw_json
import os
import json

# JSON 파일이 저장된 디렉터리 경로
json_dir = '../dataset/2_raw_json'

# 변환된 JSON 파일을 저장할 디렉터리 경로
output_dir = '../dataset/2_raw_json'
os.makedirs(output_dir, exist_ok=True)

# 변환된 파일 수 카운트
converted_count = 0

for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(json_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # 불필요한 필드 제거
            data.pop('lineColor', None)
            data.pop('fillColor', None)
            data.pop('imageData', None)  # 필요에 따라 제거
            
            # 'shapes' 내의 각 객체에서 불필요한 필드 제거
            for shape in data.get('shapes', []):
                shape.pop('lineColor', None)
                shape.pop('fillColor', None)
                shape.pop('flags', None)  # 필요에 따라 제거
            
            # 변환된 JSON 파일 저장
            output_filepath = os.path.join(output_dir, filename)
            with open(output_filepath, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
            
            print(f"✅ {filename} 변환 완료")
            converted_count += 1
        
        except json.JSONDecodeError:
            print(f"❌ JSON 파싱 오류: {filename}")
        except Exception as e:
            print(f"❌ 오류 발생: {filename} - {e}")

print(f"\n✅ 총 {converted_count}개의 JSON 파일이 변환되었습니다.")
