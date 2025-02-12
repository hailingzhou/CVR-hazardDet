from nltk.tokenize import word_tokenize
import json
import h5py
import Constants
import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from torch import nn

class HD_datasets(Dataset):
    def __init__(self, **args):
        self.mode = args['mode']
        self.split = args['split']
        if args['forbidden'] != '':
            with open(args['forbidden'], 'r') as f:
                self.forbidden = json.load(f)
            self.forbidden = set(self.forbidden)
        else:
            self.forbidden = set([])
        

        if self.split == 'balanced_train':
            with open('../datasets/hazard_detection/image_rule_pair/Rule124_triplet_balanced_train_inputs.json') as f:
                self.data = json.load(f)
            print("loading data from {}".format(
                '../datasets/hazard_detection/image_rule_pair/Rule124_triplet_balanced_train_inputs.json'))
        elif self.split == 'balanced_test':
            with open('../datasets/hazard_detection/image_rule_pair/Rule124_triplet_balanced_test_inputs.json') as f:
                self.data = json.load(f)
            print("loading data from {}".format(
                '../datasets/hazard_detection/image_rule_pair/Rule124_triplet_balanced_test_inputs.json'))
        else:
            with open('../datasets/hazard_detection/image_rule_pair/Rule124_triplet_50_{}_inputs.json'.format(self.split), 'r') as f:
                self.data = json.load(f)
            print("loading data from {}".format(
                '../datasets/hazard_detection/image_rule_pair/Rule124_triplet_50_{}_inputs.json'.format(self.split)))

        # with open('/mntnfs/med_data5/shunlinlu/mmn/meta_info/gqa_objects_merged_info.json') as f:
        #     self.object_info = json.load(f)
        # import pdb; pdb.set_trace()
        # database = set(self.object_info.keys())

        # self.data = list(filter(lambda x: x[0] in database, self.data))
        print("there are in total {} instances before validation removal".format(len(self.data)))

        self.data = list(filter(lambda x: x[-2] not in self.forbidden, self.data)) # remove those forbidden
        print("there are in total {} instances".format(len(self.data)))

        self.vocab = args['vocab']
        self.answer_vocab = args['answer']
        self.num_tokens = args['num_tokens']
        self.num_regions = args['num_regions']
        self.LENGTH = args['length']
        self.MAX_LAYER = args['max_layer']

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class HD_datasets_v2(HD_datasets):
    def __init__(self, **args):
        super(HD_datasets_v2, self).__init__(**args)
        self.folder = args['folder']
        self.threshold = args['threshold']
        self.contained_weight = args['contained_weight']
        self.cutoff = args['cutoff']
        self.distribution = args['distribution']

    def __getitem__(self, index):
        entry = self.data[index]
        # obj_info = self.object_info[entry[0]]
        # if not entry[0].startswith('n'):
        #     if len(entry[0]) < 7:
        #         entry[0] = "0" * (7 - len(entry[0])) + entry[0]

        image_id = entry[0]    # e.g. n51002
        question = entry[1]    # e.g. What is the device behind the keyboard made of plastic ?
        inputs = entry[3]      # e.g. [['select', None, None, None, 'woman', None, None, None], ['relate_name', None, None, 'staring at', 'animal', None, None, None], ['query_n', None, None, None, None, None, None, None]]
        connection = entry[4]  # e.g. [[[0, 0]], [[1, 0]], [[2, 1]], [[3, 2]]]
        questionId = entry[-2] # e.g. 20226381
        # print('im_id', image_id, 'input shape', inputs)
        # print('q_id', questionId)

        length = min(len(inputs), self.LENGTH)

        # Prepare Question
        idxs = word_tokenize(question)[:self.num_tokens]
        question = [self.vocab.get(_, Constants.UNK) for _ in idxs]
        question += [Constants.PAD] * (self.num_tokens - len(idxs))
        question = np.array(question, 'int64')


        question_masks = np.zeros((len(question), ), 'float32')
        question_masks[:len(idxs)] = 1.

        # Prepare Program
        program = np.zeros((self.LENGTH, 8), 'int64')
        depth = np.zeros((self.LENGTH, ), 'int64')
        for i in range(length):
            for j, text in enumerate(inputs[i]):
                if text is not None:
                    program[i][j] = self.vocab.get(text, Constants.UNK)

        # Prepare Program mask
        program_masks = np.zeros((self.LENGTH, ), 'float32')
        program_masks[:length] = 1.

        # Prepare Program Transition Mask
        transition_masks = np.zeros(
            (self.MAX_LAYER, self.LENGTH, self.LENGTH), 'uint8')
        activate_mask = np.zeros((self.MAX_LAYER, self.LENGTH), 'float32') # 5 x 9

        """
          [['select', None, None, None, 'table', None, None, None],
           ['relate_inv_name', None, None, 'on top of', 'utensil', None, None, None],
           ['verify', None, 'black', None, None, None, None, None],
           ['verify', None, 'clean', None, None, None, None, None],
           ['and', None, None, None, None, None, None, None]],
          [[[0, 0]], [[1, 0]], [[2, 1], [3, 1]], [[4, 2], [4, 3]]],
        """
        for i in range(self.MAX_LAYER):
            if i < len(connection):
                for idx, idy in connection[i]:
                    transition_masks[i][idx][idy] = 1 # idx: subject idy: dependency  # which functions should be activated as dependency
                    depth[idx] = i
                    activate_mask[i][idx] = 1 # when to activate a function # which functions should be activated for execution
            for j in range(self.LENGTH):
                if activate_mask[i][j] == 0:
                    # As a placeholder
                    transition_masks[i][j][j] = 1
                else:
                    pass

        vis_mask = np.zeros((self.num_regions, ), 'float32')
        # Prepare Vision Feature
        # bottom_up = np.load(os.path.join(
        #     self.folder, '{}.npz'.format(image_id)), allow_pickle=True)
        # bottom_up = np.load(os.path.join(
        #     self.folder, 'gqa_{}.npz'.format(image_id)))

        bottom_up = np.load(os.path.join(
            self.folder, '{}.npz'.format(image_id)))
        adaptive_num_regions = min(
            (bottom_up['conf'] > self.threshold).sum(), self.num_regions)


        # adaptive_num_regions = min((np.max(bottom_up['info'].tolist()['cls_prob'], axis=1) > self.threshold).sum(), self.num_regions)

        # Cut off the bottom up features
        object_feat = bottom_up['features'][:adaptive_num_regions]
        bbox_feat = bottom_up['bbox'][:adaptive_num_regions]
        # object_feat = bottom_up['features'][:adaptive_num_regions]
        # bbox_feat = bottom_up['bbox'][:adaptive_num_regions]
        vis_mask[:bbox_feat.shape[0]] = 1.
        # Padding zero
        if object_feat.shape[0] < self.num_regions:
            padding = self.num_regions - object_feat.shape[0]
            object_feat = np.concatenate([object_feat, np.zeros(
                (padding, object_feat.shape[1]), 'float32')], 0)
        if bbox_feat.shape[0] < self.num_regions:
            padding = self.num_regions - bbox_feat.shape[0]
            bbox_feat = np.concatenate([bbox_feat, np.zeros(
                (padding, bbox_feat.shape[1]), 'float32')], 0)
        num_regions = bbox_feat.shape[0]

        # exist = np.full((self.LENGTH, ), -1, 'float32')
        # if self.mode == 'train':
        #     returns = entry[2]
        #     intermediate_idx = np.full(
        #         (self.LENGTH, num_regions + 1), 0, 'float32')
        #     intersect_iou = np.full(
        #         (length - 1, num_regions + 1), 0., 'float32')
        #     for idx in range(length - 1):
        #         if isinstance(returns[idx], list):
        #             if returns[idx] == [-1, -1, -1, -1]:
        #                 intermediate_idx[idx][num_regions] = 1
        #             else:
        #                 gt_coordinate = (returns[idx][0] / (obj_info['width'] + 0.),
        #                                  returns[idx][1] / (obj_info['height'] + 0.),
        #                                  (returns[idx][2] + returns[idx][0]) / (obj_info['width'] + 0.),
        #                                  (returns[idx][3] + returns[idx][1]) / (obj_info['height'] + 0.))
        #                 for i in range(num_regions):
        #                     intersect, contain = Constants.intersect(
        #                         gt_coordinate, bbox_feat[i, :4], True, 'x1y1x2y2')
        #                     intersect_iou[idx][i] = intersect  # + self.contained_weight * contain

        #                 # if self.distribution:
        #                     #mask = (intersect_iou[idx] > self.cutoff).astype('float32')
        #                     #intersect_iou[idx] *= mask
        #                 intermediate_idx[idx] = intersect_iou[idx] / (intersect_iou[idx].sum() + 0.001)
        #                 # else:
        #                 #    intermediate_idx[idx] = (intersect_iou[idx] > self.cutoff).astype('float32')
        #                 #    intermediate_idx[idx] = intermediate_idx[idx] / (intermediate_idx[idx].sum() + 0.001)

        # else:
        intermediate_idx = 0

        # print(bbox_feat.shape)
        # Prepare index selection
        index = length - 1
        # Prepare answer
        answer_id = self.answer_vocab.get(entry[-1], Constants.UNK)
        # print(type(question))
        # import pdb; pdb.set_trace()
        return question, question_masks, program, program_masks, transition_masks, activate_mask, object_feat, \
            bbox_feat, vis_mask, index, depth, intermediate_idx, answer_id, questionId


class BCEWithMask(nn.Module):
    def __init__(self, ignore_index):
        super(BCEWithMask, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, gt):
        mask = (gt != self.ignore_index).float()
        prob = torch.sigmoid(logits)
        loss = -torch.log(prob) * gt + (-torch.log(1 - prob)) * (1 - gt)
        length = torch.sum(mask)
        loss = torch.sum(loss * mask) / length
        return loss


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, prob, logits):
        length = (prob.sum(-1) > 0.001).sum()
        pred_prob = torch.softmax(logits, -1)
        loss = -prob * torch.log(pred_prob)
        loss = torch.sum(loss, -1)
        loss = torch.sum(loss) / length
        return loss
    
    
class single_image_text_test():
    def __init__(self, **args):
        if args['forbidden'] != '':
            with open(args['forbidden'], 'r') as f:
                self.forbidden = json.load(f)
            self.forbidden = set(self.forbidden)
        else:
            self.forbidden = set([])
            
        self.LENGTH = args['length']
        self.num_tokens = args['num_tokens']
        self.vocab = args['vocab']
        self.MAX_LAYER = args['max_layer']
        self.num_regions = args['num_regions']
        self.folder = args['folder']
        self.threshold = args['threshold']
        self.answer_vocab = args['answer']
            
        with open('../datasets/hazard_detection/image_rule_pair/Rule124_rule_triplet_pair.json') as f:
            self.rule_triplet_pair = json.load(f)
                


    def forward(self, image, txt):
        entry = self._create_entry(image, txt)
        
        image_id = entry[0]
        question = entry[1]
        inputs = entry[3]
        connection = entry[4]
        questionId = entry[-2]
        
        length = min(len(inputs), self.LENGTH)

        # Prepare Question
        idxs = word_tokenize(question)[:self.num_tokens]
        question = [self.vocab.get(_, Constants.UNK) for _ in idxs]
        question += [Constants.PAD] * (self.num_tokens - len(idxs))
        question = np.array(question, 'int64')

        question_masks = np.zeros((len(question), ), 'float32')
        question_masks[:len(idxs)] = 1.

        # Prepare Program
        program = np.zeros((self.LENGTH, 8), 'int64')
        depth = np.zeros((self.LENGTH, ), 'int64')
        for i in range(length):
            for j, text in enumerate(inputs[i]):
                if text is not None:
                    program[i][j] = self.vocab.get(text, Constants.UNK)

        # Prepare Program mask
        program_masks = np.zeros((self.LENGTH, ), 'float32')
        program_masks[:length] = 1.

        # Prepare Program Transition Mask
        transition_masks = np.zeros(
            (self.MAX_LAYER, self.LENGTH, self.LENGTH), 'uint8')
        activate_mask = np.zeros((self.MAX_LAYER, self.LENGTH), 'float32') # 5 x 9

        """
          [['select', None, None, None, 'table', None, None, None],
           ['relate_inv_name', None, None, 'on top of', 'utensil', None, None, None],
           ['verify', None, 'black', None, None, None, None, None],
           ['verify', None, 'clean', None, None, None, None, None],
           ['and', None, None, None, None, None, None, None]],
          [[[0, 0]], [[1, 0]], [[2, 1], [3, 1]], [[4, 2], [4, 3]]],
        """
        for i in range(self.MAX_LAYER):
            if i < len(connection):
                for idx, idy in connection[i]:
                    transition_masks[i][idx][idy] = 1 # idx: subject idy: dependency  # which functions should be activated as dependency
                    depth[idx] = i
                    activate_mask[i][idx] = 1 # when to activate a function # which functions should be activated for execution
            for j in range(self.LENGTH):
                if activate_mask[i][j] == 0:
                    # As a placeholder
                    transition_masks[i][j][j] = 1
                else:
                    pass

        vis_mask = np.zeros((self.num_regions, ), 'float32')
        # Prepare Vision Feature
        # bottom_up = np.load(os.path.join(
        #     self.folder, '{}.npz'.format(image_id)), allow_pickle=True)
        # bottom_up = np.load(os.path.join(
        #     self.folder, 'gqa_{}.npz'.format(image_id)))

        bottom_up = np.load(os.path.join(
            self.folder, '{}.npz'.format(image_id)))
        adaptive_num_regions = min(
            (bottom_up['conf'] > self.threshold).sum(), self.num_regions)


        # adaptive_num_regions = min((np.max(bottom_up['info'].tolist()['cls_prob'], axis=1) > self.threshold).sum(), self.num_regions)

        # Cut off the bottom up features
        object_feat = bottom_up['features'][:adaptive_num_regions]
        bbox_feat = bottom_up['bbox'][:adaptive_num_regions]
        # object_feat = bottom_up['features'][:adaptive_num_regions]
        # bbox_feat = bottom_up['bbox'][:adaptive_num_regions]
        vis_mask[:bbox_feat.shape[0]] = 1.
        # Padding zero
        if object_feat.shape[0] < self.num_regions:
            padding = self.num_regions - object_feat.shape[0]
            object_feat = np.concatenate([object_feat, np.zeros(
                (padding, object_feat.shape[1]), 'float32')], 0)
        if bbox_feat.shape[0] < self.num_regions:
            padding = self.num_regions - bbox_feat.shape[0]
            bbox_feat = np.concatenate([bbox_feat, np.zeros(
                (padding, bbox_feat.shape[1]), 'float32')], 0)
        num_regions = bbox_feat.shape[0]

        # exist = np.full((self.LENGTH, ), -1, 'float32')
        # if self.mode == 'train':
        #     returns = entry[2]
        #     intermediate_idx = np.full(
        #         (self.LENGTH, num_regions + 1), 0, 'float32')
        #     intersect_iou = np.full(
        #         (length - 1, num_regions + 1), 0., 'float32')
        #     for idx in range(length - 1):
        #         if isinstance(returns[idx], list):
        #             if returns[idx] == [-1, -1, -1, -1]:
        #                 intermediate_idx[idx][num_regions] = 1
        #             else:
        #                 gt_coordinate = (returns[idx][0] / (obj_info['width'] + 0.),
        #                                  returns[idx][1] / (obj_info['height'] + 0.),
        #                                  (returns[idx][2] + returns[idx][0]) / (obj_info['width'] + 0.),
        #                                  (returns[idx][3] + returns[idx][1]) / (obj_info['height'] + 0.))
        #                 for i in range(num_regions):
        #                     intersect, contain = Constants.intersect(
        #                         gt_coordinate, bbox_feat[i, :4], True, 'x1y1x2y2')
        #                     intersect_iou[idx][i] = intersect  # + self.contained_weight * contain

        #                 # if self.distribution:
        #                     #mask = (intersect_iou[idx] > self.cutoff).astype('float32')
        #                     #intersect_iou[idx] *= mask
        #                 intermediate_idx[idx] = intersect_iou[idx] / (intersect_iou[idx].sum() + 0.001)
        #                 # else:
        #                 #    intermediate_idx[idx] = (intersect_iou[idx] > self.cutoff).astype('float32')
        #                 #    intermediate_idx[idx] = intermediate_idx[idx] / (intermediate_idx[idx].sum() + 0.001)

        # else:
        intermediate_idx = 0

        # print(bbox_feat.shape)
        # Prepare index selection
        index = length - 1
        # Prepare answer
        answer_id = self.answer_vocab.get(entry[-1], Constants.UNK)

        return question, question_masks, program, program_masks, transition_masks, activate_mask, object_feat, \
            bbox_feat, vis_mask, index, depth, intermediate_idx, answer_id, questionId
        
        
    def _find_all_nums(self, strings):
        # print(strings)
        # import pdb; pdb.set_trace()
        nums = []
        for s in strings:
            if '[' in s and ']' in s:
                nums.append(int(s[1:2]))

        return nums
        
        
    def _create_entry(self, image, txt):
        
        
        ImageId = image.split('.')[0]
        
        programs,rule_type = self._create_program(txt)
        rounds = []
        depth = {}
        cur_depth = 0
        tmp = []
        connection = []
        inputs = []
        returns = []
        tmp_connection = []
        for i, program in enumerate(programs):
            # import pdb; pdb.set_trace()
            if isinstance(program, list):
                _, func, args = Constants.parse_program(program[1])
                returns.append(program[0])
                # print(args)
            else:
                _, func, args = Constants.parse_program(program)
                # print(args)
            try:
                if func == 'relate' or func == 'relate_inv':
                    inputs.append([func, None, None, args[1], None, None, None, None])
                elif func == 'relate_attr':
                    inputs.append([func, None, None, args[1], args[2], None, None, None])
                elif func == 'relate_name' or func == 'relate_inv_name':
                    inputs.append([func, None, None, args[1], args[2], None, None, None])
                elif func == 'select':
                    inputs.append([func, None, None, None, args[0], None, None, None])
                elif func == 'filter' or func == 'filter_not':
                    inputs.append([func, None, args[1], None, None, None, None, None])
                elif func == 'filter_h' or func == 'filter_v':
                    inputs.append([func, None, None, None, None, args[1], None, None])
                elif func == 'verify_h' or func == 'verify_v':
                    inputs.append([func, None, None, None, None, args[0], None, None])
                elif func == 'query_n':
                    inputs.append([func, None, None, None, None, None, None, None])
                elif func == 'query_h' or func == 'query_v':
                    inputs.append([func, None, None, None, None, None, None, None])
                elif func == 'query':
                    inputs.append([func, args[1], None, None, None, None, None, None])
                elif func == 'query_f':
                    inputs.append([func, args[0], None, None, None, None, None, None])
                elif func == 'verify':
                    inputs.append([func, None, args[1], None, None, None, None, None])
                elif func == 'verify_f':
                    inputs.append([func, None, args[0], None, None, None, None, None])
                elif func == 'verify_rel' or func == 'verify_rel_inv':
                    inputs.append([func, None, None, args[1], args[2], None, None, None])
                elif func in ['choose_n', 'choose_h', 'choose_v']:
                    inputs.append([func, None, None, None, None, None, args[1], args[2]])
                elif func == 'choose':
                    inputs.append([func, None, None, None, None, None, args[1], args[2]])
                elif func == 'choose_subj':
                    inputs.append([func, None, args[2], None, None, None, None, None])
                elif func == 'choose_attr':
                    inputs.append([func, args[1], None, None, None, None, args[2], args[3]])
                elif func == 'choose_f':
                    inputs.append([func, None, None, None, None, None, args[0], args[1]])
                elif func == 'choose_rel_inv':
                    inputs.append([func, None, None, None, args[1], None, args[2], args[3]])
                elif func in ['same_attr', 'different_attr']:
                    inputs.append([func, None, args[2], None, None, None, None, None])
                elif func in ['exist', 'or', 'and', 'different', 'same', 'common']:
                    inputs.append([func, None, None, None, None, None, None, None])
                else:
                    raise ValueError('unknown function {}'.format(func))
            except Exception:
                print(program)
                inputs.append([func, None, None, None, None, None, None, None])

            assert len(inputs[-1]) == 8


            if len(self._find_all_nums(args)) == 0:
                tmp.append(program)
                depth[i] = cur_depth
                tmp_connection.append([i, i])



        connection.append(tmp_connection)
        cur_depth += 1
        rounds.append(tmp)

        while len(depth) < len(programs):
            tmp = []
            tmp_depth = {}
            tmp_connection = []
            for i, program in enumerate(programs):
                if i in depth:
                    continue
                else:
                    if isinstance(program, list):
                        _, func, args = Constants.parse_program(program[1])
                    else:
                        _, func, args = Constants.parse_program(program)


                    reliance = self._find_all_nums(args)

                    if all([_ in depth for _ in reliance]):
                        tmp.append(program)
                        tmp_depth[i] = cur_depth
                        for r in reliance:
                            if r > i:
                                r = i - 1
                            tmp_connection.append([i, r])
                    else:
                        continue

            if len(tmp_depth) == 0 and len(tmp) == 0 and len(tmp_connection) == 0:
                break
            else:
                connection.append(tmp_connection)
                rounds.append(tmp)
                cur_depth += 1
                depth.update(tmp_depth)
        # import pdb; pdb.set_trace()
        # results.append([entry[0], entry[1], returns, inputs, connection, entry[-2], entry[-1]])
        
        return [ImageId, txt, returns, inputs, connection, rule_type, None]
    
    def _create_program(self, txt):
        assert txt in self.rule_triplet_pair
        triplets = self.rule_triplet_pair[txt]
        
        if len(triplets) == 1:
            program_list = []
            program_0 = 'select(' + triplets[0] + ')'
            program_1 = 'exist([1])'
            program_list.append(program_0)
            program_list.append(program_1)
            return program_list, '1'
        elif len(triplets) == 3:
            #select(ground);relate_inv_name([0],sitting on,worker);exist([1])
            program_list = []
    #         program = 'select(' + triplets[2] + ');relate_inv_name([0],' + triplets[1] + ',' + triplets[0] + ');exist([1])'
            program_0 = 'select(' + triplets[2] + ')'
            program_1 = 'relate_inv_name([0],' + triplets[1] + ',' + triplets[0] + ')'
            program_2 = 'exist([1])'
            program_list = [program_0, program_1, program_2]
            return program_list, '2'
        elif len(triplets) == 2:
            program_list = []
            program_0 = 'select(' + triplets[0][2] + ')'
            program_1 = 'relate_inv_name([0],' + triplets[0][1] + ',' + triplets[0][0] + ');'
            program_2 = 'exist([1]);'
            program_3 = 'relate_inv_name([0],' + triplets[1][1] + ',' + triplets[0][0] + ');'
            program_4 = 'exist([3]);'
            program_5 = 'or([2],[4]);'
            program_list = [program_0, program_1, program_2, program_3, program_4, program_5]
            return program_list, '4'
        
        
            
            

        
        

    