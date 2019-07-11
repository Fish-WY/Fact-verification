
from pprint import pprint
import os.path
import json
from lupyne_retrieval_predict import *

import time


data_dir = './input_data'

def addSentenceToDataset(fileName):
    '''
    原数据集只有 docID 和 sentID 没有句子内容
    检索句子内容并将其加入 json 文件以备 bert 使用
    '''

    input_file_path = os.path.join(data_dir, fileName+'.json')
    with open(input_file_path, 'r') as f:
        input = json.loads(f.read())

    for key, value in input.items():
        evidence_list = value['evidence']
        for index, evidence in enumerate(evidence_list):
            hit = getSentbyID(evidence[0],evidence[1])
            if hit:
                evidence.append(hit['content'])
        if value['label'] == 'NOT ENOUGH INFO':
            # retrivel doc
            text = value['claim']
            hits = sentSearch(text)
            for hit in hits[:3]:
                evidence_list.append([hit['doc'],hit['id'],hit['content']])

    output_file_path = os.path.join(data_dir, fileName+'_content.json')
    with open(output_file_path, 'w') as f:
        json.dump(input, f, indent=2)

def createTestsetFrom(fileName):
    '''
    同上 插入的为自己检索的相关语句
    '''
    input_file_path = os.path.join(data_dir, fileName + '.json')
    with open(input_file_path, 'r') as f:
        input = json.loads(f.read())

    for key, value in input.items():
        # IR
        text = value['claim']
        hits = sentSearch(text)
        evidence_list = []
        for hit in hits[:3]:
            evidence_list.append([hit['doc'], hit['id'], hit['content']])
        value['evidence'] = evidence_list

    output_file_path = os.path.join(data_dir, 'test_content.json')
    with open(output_file_path, 'w') as f:
        json.dump(input, f, indent=2)

if __name__ == '__main__':
    time_start = time.time()

    addSentenceToDataset('train')
    print('***** train_content dumped *****')
    addSentenceToDataset('devset')
    print('***** devset_content dumped *****')
    createTestsetFrom('devset')
    print('***** test_content dumped *****')
    time_end = time.time()
    print('time cost', (time_end - time_start) / 60, 'min')
