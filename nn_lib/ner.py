
##NER labels, BIO
label2id = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }
id2label = {id:label for label,id in label2id.items()}

#读取用于训练的语料库
def read_ner_corpus(corpus_path):
    data = []
    with open(corpus_path, encoding='utf-8') as f:
        lines = f.readlines()
    sent_, label_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            label_.append(label)
        else:
            data.append((sent_, label_))
            sent_, label_ = [], []
    return data
