from dotenv import load_dotenv
import os
import pandas as pd
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import faiss
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# ✅ FastAPI 인스턴스 생성
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 허용할 출처 목록
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# ✅ FAISS 인덱스 파일 경로 설정
faiss_file_path = f"faiss_index_{datetime.now().strftime('%Y%m%d')}.faiss"

# ✅ POST 요청을 위한 데이터 모델 정의
class QueryRequest(BaseModel):
    query: str

# ✅ 엑셀 데이터 로드 및 변환 (공백 제거)
def load_excel_to_texts(file_path):
    try:
        data = pd.read_excel(file_path)
        data.columns = data.columns.str.strip()
        texts = [" | ".join([f"{col}: {row[col]}" for col in data.columns]) for _, row in data.iterrows()]
        return texts, data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"엑셀 파일 로드 오류: {str(e)}")

# ✅ FAISS 인덱스 저장
def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

# ✅ FAISS 인덱스 로드
def load_faiss_index(file_path):
    try:
        return faiss.read_index(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS 인덱스 로딩 오류: {str(e)}")

# ✅ FAISS 인덱스 생성 및 저장
def create_and_save_faiss_index(file_path):
    texts, _ = load_excel_to_texts(file_path)
    임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
    embeddings = 임베딩.embed_documents(texts)
    index = faiss.IndexFlatL2(1536)
    index.add(np.array(embeddings, dtype=np.float32))
    save_faiss_index(index, faiss_file_path)

# ✅ FAISS 인덱스 로드 또는 생성
if not os.path.exists(faiss_file_path):
    create_and_save_faiss_index("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")
index = load_faiss_index(faiss_file_path)

# ✅ JSON 직렬화를 위한 int 변환 함수
def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    return obj

# ✅ 루트 경로 - HTML 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ✅ POST 요청 처리 - `/chatbot`
@app.post("/chatbot")
def search_and_generate_response(request: QueryRequest):
    query = request.query
    print(f"🔍 사용자 검색어: {query}")  # ✅ CMD에 검색어 출력

    _, data = load_excel_to_texts("db/ownerclan_narosu_오너클랜상품리스트_OWNERCLAN_250102 필요한 내용만.xlsx")

    # ✅ OpenAI 임베딩 생성
    임베딩 = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=API_KEY)
    query_embedding = 임베딩.embed_query(query)

    # ✅ FAISS 검색 수행
    D, I = index.search(np.array([query_embedding], dtype=np.float32), k=5)

    # ✅ 검색 결과 JSON 변환
    results = []
    for idx in I[0]:
        if idx >= len(data):
            continue
        result_row = data.iloc[idx]
        result_info = {
            "상품코드": str(result_row["상품코드"]),
            "원본상품명": result_row["원본상품명"],
            "오너클랜판매가": convert_to_serializable(result_row["오너클랜판매가"]),
            "배송비": convert_to_serializable(result_row["배송비"]),
            "이미지중": result_row["이미지중"],
            "원산지": result_row["원산지"]
        }
        results.append(result_info)

    # ✅ CMD에 검색 결과 출력
    print(f"\n✅ 검색 결과 (상위 {len(results)}개):")
    for result in results:
        print(json.dumps(result, indent=4, ensure_ascii=False))

    # ✅ LLM을 사용하여 검색 결과 설명 생성
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=API_KEY)
    response = llm.invoke(f"'{query}'에 대한 검색 결과: {json.dumps(results, ensure_ascii=False)}")

    # ✅ JSON 반환
    return {
        "query": query,
        "results": results,
        "llm_response": str(response)
    }

# ✅ FastAPI 서버 실행 (포트 고정: 5050)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
