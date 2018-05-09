import sys
from nltk import tokenize
from nltk.corpus import stopwords
from string import punctuation
from pprint import pprint

stop_words = stopwords.words('english') + list(punctuation)


def summarize(text):
    clean_text = [word for word in tokenize.word_tokenize(text) if word not in stop_words]
    df = {}
    for word in clean_text:
        if word in df:
            df[word] += 1
        else:
            df[word] = 1
    sentences = tokenize.sent_tokenize(text)
    sentence_score = []
    for sentence in sentences:
        tf = {}
        words = [word for word in tokenize.word_tokenize(sentence) if word not in stop_words]
        for word in words:
            if word in tf:
                tf[word] += 1
            else:
                tf[word] = 1
        score = 0
        for word in words:
            score += tf[word] / df[word]
        sentence_score.append((sentence, score))
    top_n = [sentence for sentence, score in sorted(sentence_score, key=lambda x: x[1], reverse=True)[:int(len(sentences)/10)]]
    pprint(top_n)
    return ' '.join([sentence for sentence in sentences if sentence in top_n])


if __name__ == '__main__':
    file_name = sys.argv[1]
    if '.txt' not in file_name:
        print('ONLY ACCEPTS .txt FILES')
        sys.exit(9)
    with open(file_name, 'r') as file:
        text = file.read().replace('\n', ' ')
    print(summarize(text))
