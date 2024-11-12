from langchain_community.vectorstores import FAISS  # (3) 검색을 위한 고속 벡터 저장소 FAISS 불러오기
from langchain_community.document_loaders import TextLoader  # (3) 텍스트 파일 로드 기능을 제공하는 모듈
from langchain_text_splitters import CharacterTextSplitter  # (2) 텍스트를 일정한 길이로 나누기 위한 분할기
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings  # (3) 텍스트 임베딩을 위한 모델 불러오기

# 1단계: 문서 로드 및 전처리
# 텍스트 로더를 사용해 appendix-keywords.txt 파일을 불러오고 문서 형태로 변환
loader = TextLoader("appendix-keywords.txt", encoding="UTF-8")  # (3) 불러올 텍스트 파일 경로 및 인코딩 설정
documents = loader.load()  # (3) 문서 데이터를 메모리에 로드

# 텍스트를 일정 크기로 나누기
# CharacterTextSplitter를 사용해 각 문서를 300자 크기의 청크(부분)로 나눕니다. 중복을 방지하기 위해 겹침 없음.
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # (2) 텍스트 분할 크기 및 겹침 설정
split_docs = text_splitter.split_documents(documents)  # (2) 문서를 청크로 나누어 저장

# 2단계: 임베딩 생성 및 벡터 저장소 구축
# HuggingFace에서 제공하는 사전 훈련된 임베딩 모델 'BAAI/bge-m3'를 사용해 각 문서 청크를 벡터화
model_name = 'BAAI/bge-m3'  # (3) 사용 모델 이름 설정
embeddings = HuggingFaceEndpointEmbeddings(model=model_name)  # (3) HuggingFace의 임베딩 모델 초기화
# FAISS 벡터 저장소 생성: 임베딩된 청크를 FAISS에 저장하여 고속 유사도 검색을 가능하게 합니다.
db = FAISS.from_documents(split_docs, embeddings)  # (3) FAISS 벡터 저장소에 벡터 데이터 저장

# 검색 방법별로 retriever 생성
# 유사도 검색
retriever_similarity = db.as_retriever(search_type="similarity")  # (3) 유사도 기반으로 검색하는 검색기 설정
# MMR 검색: 다양성 고려하여 유사도 높은 결과 반환
retriever_mmr = db.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.5})  # (3) MMR 검색기 설정
# 유사도 임계값 기반 검색: 임계값 0.7 이상의 유사도 결과만 반환
retriever_threshold = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4})  # (3) 임계값 검색기 설정

# 3단계: 검색 결과를 보기 쉽게 출력하는 도우미 함수
# 검색된 문서 목록을 순서대로 출력하여 사용자가 쉽게 확인할 수 있도록 형식화
def pretty_print_docs(docs):  # (2) 검색 결과 문서 출력 함수 정의
    for i, doc in enumerate(docs):
        print(f"Keyword {i+1}: {doc.page_content}")  # 각 문서의 내용을 순서대로 출력
        print("-" * 50)  # 가독성을 위한 구분선 출력

# 4단계: 통합 질문-답변 시스템
# 다양한 검색 방식으로 사용자의 질문에 대한 답변을 검색하는 시스템
def interactive_qa_system():  # (2) 사용자와 상호작용하는 질문-답변 시스템 함수 정의
    print("질문을 입력하고 검색 방식을 선택하세요:")
    query = input("질문: ")  # (2) 사용자로부터 검색 질문 입력 받기
    
    # 유사도 검색 수행
    print("\n===== 유사도 검색 결과 =====")
    docs_similarity = retriever_similarity.invoke(query)  # (3) 유사도 검색을 통해 유사한 문서 반환
    pretty_print_docs(docs_similarity)  # (2) 검색 결과 출력

    # MMR 검색 수행
    print("\n===== MMR 검색 결과 =====")
    docs_mmr = retriever_mmr.invoke(query)  # (3) MMR 검색을 통해 다양성이 고려된 결과 반환
    pretty_print_docs(docs_mmr)  # (2) 검색 결과 출력

    # 유사도 임계값 기반 검색 수행
    print("\n===== 임계값 검색 결과 =====")
    docs_threshold = retriever_threshold.invoke(query)  # (3) 임계값을 충족하는 문서만 반환
    pretty_print_docs(docs_threshold)  # (2) 검색 결과 출력

# 통합 질문-답변 시스템 실행
interactive_qa_system()  # (2) 함수 호출하여 검색 시스템 실행