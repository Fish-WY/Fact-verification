from allennlp.predictors.predictor import Predictor
from pprint import pprint
predictor_fact = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")

predictor_ner = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")

# predictor_pos = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")


#'words' 'tags'

print('predictor loaded...')

if __name__ == '__main__':
    predictor_ner.predict(
        sentence="Did Uriah honestly think he could beat The Legend of Zelda in under three hours?"
    )

    ans = predictor_ner.predict(
        sentence="Did Uriah honestly think he could beat The Legend of Zelda in under three hours?"
    )

    # ans = predictor_pos.predict(
    #     sentence="If I bring 10 dollars tomorrow, can you buy me lunch?"
    # )
    pprint(ans)