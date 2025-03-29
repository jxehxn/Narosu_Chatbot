import os
import pandas as pd

# 엑셀 파일이 있는 폴더 경로
folder_path = './Data'  # 경로는 필요 시 수정
output_file = os.path.join(folder_path, 'filtered_merged_output.xlsx')

# 남기고 싶은 열만 지정
columns_to_keep = ['마켓상품명', '오너클랜판매가', '모델명', '배송비', '브랜드', '이미지중']

# 결과 저장용 리스트
merged_data = []

# 파일 탐색 및 필터링
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        file_path = os.path.join(folder_path, filename)
        try:
            print(f"📄 처리 중: {filename}")
            df = pd.read_excel(file_path, header=1, engine="openpyxl")

            # 필요한 열만 존재하는 경우에만 필터링 적용
            existing_cols = [col for col in columns_to_keep if col in df.columns]
            missing_cols = set(columns_to_keep) - set(existing_cols)

            # 누락된 열은 빈 값으로 추가 (열 통일)
            for col in missing_cols:
                df[col] = ''

            # 필요한 열만 선택하고 순서 정렬
            df_filtered = df[columns_to_keep]

            merged_data.append(df_filtered)
        except Exception as e:
            print(f"⚠️ {filename} 처리 중 에러 발생: {e}")

# 병합 및 저장
if merged_data:
    result_df = pd.concat(merged_data, ignore_index=True)
    result_df.to_excel(output_file, index=False)
    print(f"\n✅ 필터링된 데이터가 '{output_file}'에 저장되었습니다.")
else:
    print("❌ 병합할 데이터가 없습니다.")
