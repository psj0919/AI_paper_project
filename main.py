import json

def load_json_file(file_path):
    try:
        with open(file_path, 'rt', encoding='UTF8') as file:
            json_data = json.loads(file)
        return json_data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"Error while loading JSON file: {e}")
        return None

def main():
    # 파일 경로 상수로 정의
    file_path = 'C:/Users/82107/AI_project_dataset/data/Training/labeling_data/training_paper/training_논문/논문요약20231006_0.json'

    # JSON 파일 로드

    with open(file_path, 'rt', encoding='UTF8') as file:
        json_data = json.load(file)
    if json_data is not None:
        # 데이터 저장할 리스트
        papers = []

        # json_data[0]에 있는 각 데이터를 papers 리스트에 추가
        for paper_info in json_data:
            paper = {
                "doc_type": paper_info['data'].get("doc_type", ""),
                "doc_id": paper_info.get("doc_id", ""),
                "title": paper_info.get("title", ""),
                "date": paper_info.get("date", ""),
                "summary_text": paper_info.get("summary_text", ""),
            }
            papers.append(paper)



if __name__ == '__main__':
    main()