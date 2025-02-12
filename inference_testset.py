from torch import nn
import torch
import torch.optim as optim
import numpy as np
import argparse
import sys
import time
import os
import copy
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Networks.modular import *
# from GQA import *
from datasets import *
import glob
import resource
import itertools
from collections import Counter
import json

device = torch.device('cuda')
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40000, rlimit[1]))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max_layer', type=int, default=5, help="whether to train or test the model")
    parser.add_argument('--stacking', type=int, default=6, help="whether to train or test the model")
    parser.add_argument('--visual_dim', type=int, default=2048, help="whether to train or test the model")
    parser.add_argument('--additional_dim', type=int, default=4, help="whether to train or test the model")
    parser.add_argument('--hidden_dim', type=int, default=512, help="whether to train or test the model")
    parser.add_argument('--n_head', type=int, default=8, help="whether to train or test the model")
    
    parser.add_argument('--contained_weight', type=float, default=0.1, help="whether to train or test the model")
    parser.add_argument('--threshold', type=float, default=0., help="whether to train or test the model")
    parser.add_argument('--cutoff', type=float, default=0.5, help="whether to train or test the model")
    parser.add_argument('--dropout', type=float, default=0.1, help="whether to train or test the model")
    
    parser.add_argument('--length', type=int, default=9, help="whether to train or test the model")
    parser.add_argument('--meta', default="meta_info/", type=str, help="The hidden size of the state")
    parser.add_argument('--object_info', type=str, default='gqa_objects_merged_info.json', help="whether to train or test the model")
    parser.add_argument('--num_regions', type=int, default=48, help="whether to train or test the model")
    parser.add_argument('--distribution', default=False, action='store_true', help="whether to train or test the model")
    parser.add_argument('--num_tokens', type=int, default=30, help="whether to train or test the model")
    parser.add_argument('--pre_layers', type=int, default=3, help="whether to train or test the model")
    parser.add_argument('--intermediate_layer', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--word_glove', type=str, default="meta_info/en_emb.npy",
                        help="whether to train or test the model")
    parser.add_argument('--single', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--load_from', type=str, default="", help="whether to train or test the model")
    parser.add_argument('--data', type=str, default="/mntnfs/med_data5/shunlinlu/datasets/hazard_detection/roi_feats_after_processing",
                        help="whether to train or test the model")
    
    parser.add_argument('--forbidden', default="", type=str, help="The hidden size of the state")
    parser.add_argument('--batch_size', type=int, default=256, help="whether to train or test the model")
    parser.add_argument('--num_workers', type=int, default=16, help="whether to train or test the model")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_opt()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    print(args)

    with open('/mntnfs/med_data5/shunlinlu/hazard_detection/meta_info/full_vocab.json', 'r') as f:
        vocab = json.load(f)
        ivocab = {v: k for k, v in vocab.items()}


    with open('/mntnfs/med_data5/shunlinlu/hazard_detection/meta_info/answer_vocab.json', 'r') as f:
        answer = json.load(f)
        inv_answer = {v: k for k, v in answer.items()}

    MAX_LAYER = args.max_layer

    model = TreeTransformerSparsePostv2(vocab_size=len(vocab), stacking=args.stacking, answer_size=len(answer), visual_dim=args.visual_dim,
                                        coordinate_dim=args.additional_dim, hidden_dim=args.hidden_dim, n_head=args.n_head, n_layers=MAX_LAYER,
                                        dropout=args.dropout, intermediate_dim=args.num_regions + 1, pre_layers=args.pre_layers,
                                        intermediate_layer=args.intermediate_layer)
    print("Running Modular Transformer model with {} layers with post layer".format(args.stacking))

    model.embedding.weight.data.copy_(torch.from_numpy(np.load(args.word_glove)))
    print("loading embedding from {}".format(args.word_glove))

    if not args.single:
        model = nn.DataParallel(model)
    model.to(device)

    basic_kwargs = dict(length=args.length, object_info=os.path.join(args.meta, args.object_info),
                        num_regions=args.num_regions, distribution=args.distribution,
                        vocab=vocab, answer=answer, max_layer=MAX_LAYER, num_tokens=args.num_tokens,
                        spatial_info='{}/gqa_spatial_merged_info.json'.format(args.meta),
                        forbidden=args.forbidden)


    model.load_state_dict(torch.load(args.load_from))

    testdev_split = 'balanced_test'
    test_dataset = HD_datasets_v2(split=testdev_split, mode='val', contained_weight=args.contained_weight,
                            threshold=args.threshold, folder=args.data, cutoff=args.cutoff, **basic_kwargs)
    
    
    
    if args.num_workers == 1:
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers)
    
    
    model.eval()
    success, total = 0, 0
    for i, batch in enumerate(test_dataloader):
        
        questionId = batch[-1]
        batch = tuple(Variable(t).to(device) for t in batch[:-1])
        import pdb; pdb.set_trace()
        results = model(*batch[:-2])
        if isinstance(results, tuple):
            logits = results[1]
        else:
            logits = results
            
        preds = torch.argmax(logits, -1)
        success_or_not = (preds == batch[-1]).float()

        success += torch.sum(success_or_not).item()
        total += success_or_not.size(0)
    
    acc = round(success / (total + 0.), 4)
    print("accuracy = {}".format(acc))

    
    

    




















