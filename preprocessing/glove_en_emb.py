
import json
import numpy as np
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model

# emb = loadGloveModel('glove/glove.6B.300d.txt')
emb = loadGloveModel('../../datasets/hazard_detection/glove.6B.300d.txt')
def save(inputs, outputs):
    with open(inputs) as f:
        vocab = json.load(f)

    found, miss = 0, 0
    en_emb = np.zeros((len(vocab), 300), 'float32')
    for w, i in vocab.items():
        if w.lower() in emb:
            en_emb[i] = emb[w.lower()]
            found += 1
        elif ' ' in w:
            for w_elem in w.split(' '):
                if w_elem.lower() in emb:
                    en_emb[i] += emb[w_elem.lower()]
        else:
            print(w)
            miss += 1

    print("found = {}, miss = {}".format(found, miss))
    np.save(outputs, en_emb)

save('../../datesets/hazard_detection/meta_info/full_vocab.json', '../../datesets/hazard_detection/meta_info/en_emb.npy')