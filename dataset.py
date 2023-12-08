import json
import torch
from torch.utils.data import Dataset



class paperdata(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = self.paper_dataset()
    
    def __len__(self):
        return len(self.data)

    def paper_dataset(self):
    # 파일 경로 상수로 정의
        file_path = self.path

        # JSON 파일 로드
        with open(file_path, 'rt', encoding='UTF8') as file:
            json_data = json.load(file)
        if json_data is not None:
            # 데이터 저장할 리스트
            datas = []

            # json_data[0]에 있는 각 데이터를 papers 리스트에 추가
            for paper_info in json_data:
                for i in range(len(paper_info['data'])):
                    data = {
                        "doc_type": paper_info['data'][i].get("doc_type", ""),
                        "doc_id": paper_info['data'][i].get("doc_id", ""),
                        "title": paper_info['data'][i].get("title", ""),
                        "date": paper_info['data'][i].get("date", ""),
                        "orginal_text": paper_info['data'][i]["summary_entire"][0].get("orginal_text", ""),
                        "summary_text": paper_info['data'][i]["summary_entire"][0].get("summary_text", ""),
                    }
                    datas.append(data)
            

        return datas

    def __getitem__(self, index):
        sample = self.data[index]
        return sample['orginal_text'], sample['summary_text']



if __name__=='__main__':
    data = paperdata('/home/work/paper_data/논문요약20231006_0.json')
    
