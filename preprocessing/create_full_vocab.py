import Constants
from collections import Counter
import json

def create_vocab(input_file):
    counter = Counter()

    with open(input_file) as f:
        r = json.load(f)
    # print(len(r), r[0])
    # import pdb; pdb.set_trace()

    for i in r:
        q = i['rule']
        q = q.split(' ')
        p = i['program']
        counter.update(q)
        counter.update(p)

    print('finished reading the question-program pairs')
    vocab = {"[PAD]": Constants.PAD, "[EOS]": Constants.EOS, "[UNK]": Constants.UNK, "[SOS]": Constants.SOS}
    for (w, freq) in counter.most_common():
        vocab[w] = len(vocab)

    # with open('/mntnfs/med_data5/shunlinlu/datasets/hazard_detection/image_rule_pair/full_vocab.json', 'w') as f:
    #     json.dump(vocab, f)

    with open('../../datasets/hazard_detection/image_rule_pair/full_vocab.json', 'w') as f:
        json.dump(vocab, f)

# create_vocab('/mntnfs/med_data5/shunlinlu/datasets/hazard_detection/image_rule_pair/Rule124_triplet_50_set1_set2_answer_pairs.json')
create_vocab('../../datasets/hazard_detection/image_rule_pair/Rule124_triplet_50_set1_set2_answer_pairs.json')

