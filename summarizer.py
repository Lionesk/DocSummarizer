import sys
from nltk import tokenize
from nltk.corpus import stopwords
from string import punctuation

stop_words = stopwords.words('english') + list(punctuation)


def summarize(text):
    """
    Summarizes a text by applying tf-idf to each sentence, and selecting the top n sentences by sum of their tf-idf
    values.

    :param text: A string representation of a text
    :return: A string composed of the top sentences from the input in original reading order.
    """
    # Tokenizing the words and removing the stopwords and punctuation
    clean_text = [word for word in tokenize.word_tokenize(text) if word not in stop_words]

    # Building a document frequency (df) of each unique token in the entire text body
    df = {}
    for word in clean_text:
        if word in df:
            df[word] += 1
        else:
            df[word] = 1

    # Splitting the sentences into separate tokens
    sentences = tokenize.sent_tokenize(text)

    # Ranking each sentence by the sum of each word multiplied its term frequency inverse document frequency (tf-idf)
    # value
    sentence_score = []
    for sentence in sentences:
        # Building the frequency dict of each word within the sentence
        tf = {}
        words = [word for word in tokenize.word_tokenize(sentence) if word not in stop_words]
        for word in words:
            if word in tf:
                tf[word] += 1
            else:
                tf[word] = 1
        # The score of the sentence is the sum of its tf-idf values
        score = sum([tf[word] / df[word] for word in set(words)])
        sentence_score.append((sentence, score))

    # Selecting the top n (where n is the number of sentences divided by 10) sentences by score as they appear in the
    # text such that the reading order of the sentences is preserved
    top_n = [sentence for sentence, score in sorted(sentence_score, key=lambda x: x[1], reverse=True)[:int(len(sentences)/10)]]
    return ' '.join([sentence for sentence in sentences if sentence in top_n])


if __name__ == '__main__':
    file_name = sys.argv[1]
    if '.txt' not in file_name:
        print('ONLY ACCEPTS .txt FILES')
        sys.exit(9)
    with open(file_name, 'r') as file:
        text = file.read().replace('\n', ' ')
    print(summarize(text))
