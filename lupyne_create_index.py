import lucene
import nltk
from lupyne import engine

import time
time_start=time.time()

lucene.initVM()

# Store the index in memory:
indexer = engine.Indexer(directory='doc_sentence')              # Indexer combines Writer and Searcher; RAMDirectory and StandardAnalyzer are defaults
indexer.set('content', engine.Field.Text, stored=True)
indexer.set('sent', engine.Field.Text, stored=True)
indexer.set('doc', engine.Field.Text, stored=True)
indexer.set('id', engine.Field.Text, stored=True)

doc_indexer = engine.Indexer(directory='stemmer_doc')
doc_indexer.set('content', engine.Field.Text, stored=True)
doc_indexer.set('doc', engine.Field.Text, stored=True)
doc_indexer.set('id', engine.Field.Text, stored=True)


lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

def lemmatize(word):
    if not word.isalpha():
        return word
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def preprocessSentence(stc):
    stc = stc.lower().replace('_',' ').replace('-lrb-','( ').replace('-rrb-',' )').replace('-lsb-','[ ').replace('-rsb-',' ]').replace('``','\'').replace('\'\'','\'')
    tokenized_sentence = word_tokenizer.tokenize(stc)
    words = [stemmer.stem(token) for token in tokenized_sentence]
    return ' '.join(words)

def replaceSentece(stc):
    return stc.replace('_',' ').replace('-LRB-','( ').replace('-RRB-',' )').replace('-LSB-','[ ').replace('-RSB-',' ]').replace('``','\'').replace('\'\'','\'')


# create index for each wiki-xxx.txt
for i in range(1, 110):
    fname = 'wiki-' + format(i, '03') + '.txt'
    print(fname)

    with open('/Users/kris/Documents/UNI_PDF/Web Search and Text Analysis COMP90042_2019_SM1/project/wiki-pages-text/' + fname, 'r') as f:
        cache = 0 # 显示index 生成进度
        tmp_content = ''
        tmp_doc_id = ''
        for line in f:
            cache += 1
            if cache % 30000 == 0:
                print(cache)


            # lowecase and extract doc_id snet_id
            tokens = line.rstrip('\n').split(' ',2)
            try:
                sentence_id = int(tokens[1])
                doc_id = tokens[0]
                doc_content = tokens[2]
                #print(doc_id)
                # write sentence to  snetence dataset
                indexer.add(id=tokens[0],
                            doc=replaceSentece(tokens[0]),
                            sent=tokens[1],
                            content='( '+ replaceSentece(tokens[0]) + ' ) ' +replaceSentece(doc_content)
                            )

                # # concat sentence to document and write them as one entry to document dataset
                # if tokens[0] != tmp_doc_id:
                #     if tmp_doc_id:
                #         doc_indexer.add(id=tmp_doc_id,
                #                         doc=preprocessSentence(tmp_doc_id),
                #                         content=preprocessSentence(tmp_content)
                #                         )
                #     tmp_content = ''
                #     tmp_doc_id = doc_id
                # else:
                #     tmp_content += doc_content + ' '

            except Exception as e: # ignore sentences with wrong structure
                # print(e)
                # print(line)
                pass

        indexer.commit()
        doc_indexer.commit()



print('lupyne index finish')
print(cache)

# test data by search
hits = indexer.search('1992_Northwestern_Wildcats_football_team', field='doc')    # parsing handled if necessary
print(len(hits))
# hits support mapping interface
for hit in hits[:5]:
    print(hit)
# closing is handled automatically


time_end=time.time()
print('time cost',(time_end-time_start)/60,'min')
