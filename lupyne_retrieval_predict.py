from nltk.tag import StanfordNERTagger
import nltk
from pprint import pprint
import collections
import os.path
import json
from allennlp_models import *

import lucene
from lupyne import engine
lucene.initVM()

import time
time_start=time.time()


# stem each word in sentence
stemmer = nltk.stem.porter.PorterStemmer()
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

def preprocessSentence(stc):
    stc = stc.lower().replace('_',' ').replace('-lrb-','( ').replace('-rrb-',' )').replace('-lsb-','[ ').replace('-rsb-',' ]').replace('``','\'').replace('\'\'','\'')
    tokenized_sentence = word_tokenizer.tokenize(stc)
    words = [stemmer.stem(token) for token in tokenized_sentence]
    return ' '.join(words)

def encodeSentence(stc):
    return stc.replace('_',' ').replace('-lrb-','( ').replace('-rrb-',' )').replace('-lsb-','[ ').replace('-rsb-',' ]').replace('``','\'').replace('\'\'','\'')



indexer_doc = engine.indexers.IndexSearcher(directory='stemmer_doc')

indexer_sent = engine.indexers.IndexSearcher(directory='doc_sentence')


def getSentbyID(doc_id, sent_id):
    '''
    根据 ID 返回对应的句子内容
    '''
    try:
        sent_id = str(sent_id)
        # q = Query.term('doc', doc_id) & Query.term('sent', sent_id)
        q = '+id:' + doc_id + ' +sent:' + sent_id
        # print(q)
        hits = indexer_sent.search(q,count=3)
        #print(hits.count)
        if hits.count:
            #print(hits[0])
            return hits[0]
        return None
    except:
        print('getSentbyID error')
        return None

def docSearch(classified_text=[]):
    NER_list = []
    # tmp = []
    # combine NER words
    # for NER in classified_text:
    #     if NER[1] != 'O':
    #         tmp.append(NER[0].lower())
    #     else:
    #         if tmp:
    #             NER_list.append('doc:\"' + ' '.join(tmp) + '\"^5')
    #             tmp = []
    #         if NER[0].isalnum():
    #             NER_list.append('doc:\"' + NER[0].lower() + '\"')

    # split NER
    for NER in classified_text:
        if NER[1] == 'O':
            NER_list.append('content:\"' + NER[0].lower() + '\"')
        else:
            NER_list.append('content:\"' + NER[0].lower() + '\"^5')


    # query_str = ' OR '.join(NER_list)
    print(NER_list)

    # q1 = ['doc:\"' + query +'\"' for query in query_list]
    q = ' OR '.join(NER_list)
    print(q)
    hits = indexer_doc.search(q,count = 10)
    print(hits.count)
    # for hit in hits[:10]:
    #     print(hit)
    return hits

def sentSearch(claim = ''):
    '''
    返回相关的句子排序
    提取句子中的 NER 并给其更高的查询权重
    查询时不仅考虑句子内容也将所属文章的标题同时考虑
    '''


    # # POS tagging v2
    # POS_list = nltk.pos_tag(nltk.word_tokenize(claim))
    # q_list = []
    # print(POS_list)
    # for POS in POS_list:
    #     if POS[0].isalnum():
    #         if POS[1][0] == 'N':
    #             q_list.append('content:\"' + stemmer.stem(POS[0]) + '\"^4')
    #             q_list.append('doc:\"' + stemmer.stem(POS[0]) + '\"^4')
    #         elif POS[1][0] == 'V':
    #             q_list.append('content:\"' + stemmer.stem(POS[0]) + '\"')
    #             pass

    # POS tagging v3 CAPS doc_sentence
    POS_list = nltk.pos_tag(nltk.word_tokenize(claim))
    content_query_list = []
    #print(POS_list)
    for POS in POS_list:
        if POS[0].isalnum():
            if POS[1][0] == 'N':
                content_query_list.append('\"' + POS[0] + '\"^4')
            elif POS[1][0] == 'V':
                content_query_list.append('\"' + POS[0] + '\"')
            else:
                content_query_list.append('\"' + POS[0] + '\"')
                pass
    content_query = 'content:( ' + ' '.join(content_query_list) + ' )' if content_query_list else ''

    # # NER tagging v4 CAPS
    # ans = predictor_ner.predict(
    #     sentence=claim
    # )
    # NER_list = zip(ans['words'], ans['tags'])
    # print(NER_list)
    # for NER in NER_list:
    #     if NER[0].isalnum():
    #         if NER[1][0] != 'O':
    #             content_query_list.append('+\"' + NER[0] + '\"')
    #             # doc_query_list.append('doc:\"' + NER[0] + '\"^4')
    #         # else:
    #         #     doc_query_list.append('\"' + NER[0] + '\"')
    # content_query = 'content:( ' + ' '.join(content_query_list) + ' )' if content_query_list else ''

    # NER tagging v3 CAPS
    ans = predictor_ner.predict(
        sentence=claim
    )
    NER_list = zip(ans['words'], ans['tags'])
    doc_query_list = []
    #print(NER_list)
    for NER in NER_list:
        if NER[0].isalnum():
            if NER[1][0] != 'O':
                doc_query_list.append('\"' + NER[0] + '\"^4')
                # doc_query_list.append('doc:\"' + NER[0] + '\"^4')
            # else:
            #     doc_query_list.append('\"' + NER[0] + '\"')
    doc_query = ' doc:( ' + ' '.join(doc_query_list) +' )' if doc_query_list else ''

    q = content_query + doc_query
    #print(q)

    hits = indexer_sent.search(q,count=20)
    #print('sentSearch',hits.count)
    # for hit in hits[:10]:
    #     print(hit)
    return hits

def predictLabel(value,hits):
    '''
    基于 allenNLP 模型的预测函数
    已被替代
    '''
    print('_'*15,'predictLabel', '_'*15)
    claim = value['claim']
    value['evidence'] = []
    raw_evidence = []
    value['label'] = ''
    evidence_str = ''

    # combine all snetences to predict
    if hits:
        for hit in hits[:5]:
            print(hit)
            value['evidence'].append([hit['id'], int(hit['sent'])])
            evidence_str += hit['content']
        ans = predictor_fact.predict(
            hypothesis=claim,
            premise=evidence_str
        )
        print(ans['label_probs'])
        label_list = ['SUPPORTS','REFUTES',"NOT ENOUGH INFO"]
        index = ans['label_probs'].index(max(ans['label_probs']))
        value['label'] = label_list[index]


    # # predict per sentence
    # if hits:
    #     for hit in hits[:10]:
    #         print(hit)
    #         # first_sent = getSentbyID(hit['id'], '0')['content']
    #         # evidence_str = first_sent + hit['content'] if first_sent and hit['sent']!='0' else hit['content']
    #         evidence_str = hit['content']
    #         # print(evidence_str)
    #         ans = predictor_fact.predict(
    #             hypothesis=claim,
    #             premise=evidence_str
    #         )
    #
    #         # get the most possible result
    #         label_list = ['SUPPORTS', 'REFUTES', "NOT ENOUGH INFO"]
    #         index = ans['label_probs'].index(max(ans['label_probs']))
    #
    #         print(ans['label_probs'])
    #         print(label_list[index])
    #
    #         if index == 2:
    #             continue
    #         # value['evidence'].append([hit['id'], int(hit['sent'])])
    #
    #         if index == 0:
    #             # if value['label'] != 'SUPPORTS':
    #             #     raw_evidence = []
    #             value['label'] = 'SUPPORTS'
    #             raw_evidence.append([hit['id'], int(hit['sent']), ans['label_probs'][0]])
    #         elif index == 1:
    #             if value['label'] == 'NOT ENOUGH INFO':
    #                 value['label'] = 'REFUTES'
    #             raw_evidence.append([hit['id'], int(hit['sent']), ans['label_probs'][1]])
    #
    #         # SUPPORTS, REFUTES, NOTENOUGHINFO = ans['label_probs']
    #         # print(ans['label_probs'])
    #         # if SUPPORTS > 0.8:
    #         #     value['label'] = 'SUPPORTS'
    #         #     value['evidence'].append([hit['doc'],int(hit['sent'])])
    #         #     # return
    #         # if REFUTES > 0.8:
    #         #     value['label'] = 'REFUTES'
    #         #     value['evidence'].append([hit['doc'],int(hit['sent'])])
    #         #     # return
    # rank raw_evidence
    # raw_evidence.sort(key= lambda x : x[2], reverse=True)
    print(raw_evidence)
    value['evidence'] = [evidence[:2] for evidence in raw_evidence[:1]]
    # print(value['evidence'])
    if not value['label']:
        value['label'] = "NOT ENOUGH INFO"
    if value['label'] == "NOT ENOUGH INFO":
        value['evidence'] = []

def model_test(value):
    claim = value['claim']
    # print('claim:', claim)
    # print('Evidence:',evidence_str)
    evidence_str = ''
    for doc_id, sent_id in value['evidence']:
        evidence = getSentbyID(doc_id, str(sent_id))
        if evidence:
            evidence_str += evidence['content']
    print(evidence_str)
    ans = predictor_fact.predict(
        hypothesis=claim,
        premise=evidence_str
    )
    label_list = ['SUPPORTS','REFUTES',"NOT ENOUGH INFO"]
    index = ans['label_probs'].index(max(ans['label_probs']))
    value['label'] = label_list[index]




def saveResults():
    with open("./testoutput.json", "w") as f:
        json.dump(input_doc, f, indent=2)
    print("json dump finish ...")


if __name__ == '__main__':
    '''
    使用 AllenNLP 预测 label
    已被替代
    '''


    # with open('train.json', 'r+') as f:
    #     train = json.loads(f.read())
    with open('test-unlabelled.json', 'r+') as f:
        input_doc = json.loads(f.read())
    # with open('devset.json', 'r+') as f:
    #     input_doc = json.loads(f.read())
    # with open('devset.json', 'r+') as f:
    #     dev = json.loads(f.read())

    # # Now search the index:
    # hits = indexer.search('Fox_Broadcasting_Company',field='doc')    # parsing handled if necessary
    # print(len(hits))
    # # hits support mapping interface
    #
    # for hit in hits[:5]:
    #     print(hit)
    # # closing is handled automatically

    # q = Query.term('doc', 'wang yao') & Query.term('sent', 'he tong')
    # print(q)

    count = 0
    correct_label = 0.0
    prec = prec_hits = 1
    rec = rec_hits = 1

    for key, value in input_doc.items():
        # if value['label'] == "NOT ENOUGH INFO": continue
        count +=1
        tmp_label = value['label']

        print('_'*15,key,'_'*15)
        text = value['claim']
        pprint(value)

        # # get train evidence
        # evidence_str = ''
        # hits = []
        # for doc_id, sent_id in value['evidence']:
        #     tmp = getSentbyID(doc_id, str(sent_id))
        #     if tmp:
        #         hits.append(tmp)
        #         evidence_str += tmp['content']

        # model_test(value)


        # # NER tagging extract key words and give them high weight in search
        # print('~' * 10, 'NER tagger', '~' * 10)
        # tokenized_text = word_tokenize(text)
        # # print(tokenized_text)
        # # NERtag
        # classified_text = st.tag(tokenized_text)
        # print(classified_text)
        # # retrieval relavent doc
        # hits = docSearch(classified_text)


        # directly retrivel sentence
        hits = sentSearch(text)

        # rerank sentences
        # hits = rerankStence(hits,value)

        # score strip
        for hit in hits[:10]:
            print(hit['id'],hit['sent'],hit.score,hit['content'])

        # max_score = hits[0].score
        # # value['evidence'] = []
        # # for hit in hits[:5]:
        # #     # if (max_score - hit.score) < 20:
        # #     print(hit)
        # #     value['evidence'].append([hit['id'], int(hit['sent'])])
        # value['evidence'] = [[hit['id'],int(hit['sent'])] for hit in hits[:20]]


        # predict label
        predictLabel(value,hits)


        # # check result
        #
        # print('____check results_____')
        # print(value['label'])
        # if value['label'] == tmp_label:
        #     correct_label += 1
        #
        # aes = dev[key]['evidence']
        # pes = value['evidence']
        # for ae in aes:
        #     if ae in pes:
        #         rec += 1
        #     else:
        #         getSentbyID(ae[0],str(ae[1]))
        #     rec_hits += 1
        # print('sentence recall : {:.1f}%'.format((rec / rec_hits) * 100))
        # for pe in pes:
        #     if pe in aes:
        #         prec += 1
        #     prec_hits += 1
        # print('sentence precision : {:.1f}%'.format((prec / prec_hits) * 100))
        #
        # print('correct label rate: {:.1f}%'.format((correct_label/count)*100))
        # print()

    # write results
    saveResults()
    print('predict finish')

    time_end=time.time()
    print('time cost',(time_end-time_start)/60,'min')
