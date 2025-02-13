import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import math
from Networks.mnnm_models.gqa_mnnm_constants import *
# import numpy as np
# from general_deep_modules import FC, MLP, LayerNorm
from Networks.mnnm_models.attention_modules import *
# from meta_modules import *
from Networks.mnnm_models.bert_modules import *


class SoftLogicTransformer(nn.Module):
    def __init__(self, vocab_size, answer_size, visual_dim, coordinate_dim,
                 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers, intermediate_layer):
        super(SoftLogicTransformer, self).__init__()
        self.forward_ques = False
        self.forward_vis = False
        self.config = BertConfig(vocab_size_or_config_json_file=vocab_size)
        hidden_dim = self.config.hidden_size
        self.bert_layer = BertLayer(self.config)

        # The question encoder
        self.ques_bert_embedding = BertEmbeddings(self.config)
        self.embedding = nn.Embedding(vocab_size, 300, padding_idx=PAD)
        self.ques_proj = nn.Linear(300, hidden_dim)
        self.prog_proj = nn.Linear(300, hidden_dim // 8)
        
        # self.prog_linear = nn.Linear(300, hidden_dim)
        # self.prog_proj2 = nn.Linear(hidden_dim, hidden_dim//8)
        # self.prog_self_attention = BertSelfattLayer(self.config)
        # The visual encoder
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
        self.ques_encoder = nn.ModuleList([BertSelfattLayer(self.config) for _ in range(pre_layers)])
        self.vis_encoder = nn.ModuleList([BertCrossattLayer(self.config) for _ in range(pre_layers)])

        # The execution modules
        self.exec_self_attention = BertSelfattLayer(self.config)
        self.exec_cross_attention = BertCrossattLayer(self.config)

        # The post modules
        self.post = nn.ModuleList([BertCrossattLayer(self.config) for _ in range(stacking)])

        # The program decoder
        self.num_regions = intermediate_dim
        self.pos_emb = nn.Embedding(self.num_regions, hidden_dim)
        self.idx_predictor = nn.Linear(hidden_dim, self.num_regions)

        # Projection layer to retrieve final answer
        # self.adapted_w = nn.Parameter(torch.ones(2, dataset.num_ans_candidates))
        self.proj = nn.Linear(hidden_dim, answer_size) if not (self.forward_ques or self.forward_vis) else nn.Linear(hidden_dim * 2, answer_size)


    def forward(self, ques, ques_masks, program, program_masks, transition_masks, activate_masks, vis_feat, box_feat,
                vis_mask, index, depth, intermediate_idx=None):
        # ques: [BS, QD], ques_masks: [BS, QD], program: [BS, T, PF], program_masks: [BS, R],
        # transition_masks: [BS, LAYER, R, R], activate_masks: [BS, LAYER, R], vis_feat: [BS, N, D],
        # box_feat: [BS, N, 6], vis_mask: [BS, N], index: [BS], depth: [BS, R], intermediate_idx: [BS, R, N]
        
        # ques: [512, 30], ques_mask: [512, 30], program: [512,9,8], program_masks[512, 9]
        # transition_masks[512, 5, 9, 9], activate_masks[512, 5, 9], vis_feat [512, 48, 2048]
        # box_feat [512, 48, 6], activate_mask[512, 48], index: [512], depth: [512, 9], intermediate_idx: None
        batch_size = ques.size(0)
        idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(ques.device)
        # import pdb; pdb.set_trace()
        vis_feat = self.vis_proj(vis_feat) + self.coordinate_proj(box_feat) + self.pos_emb(idx)

        program_mask_tmp = (1 - program_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
        ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2) * (-1e9)

        # Encoding the question with self-attention
        ques_input = self.ques_bert_embedding(ques, )
        for enc in self.ques_encoder:
            ques_input = enc(ques_input, ques_mask_tmp)
            ques_input *= ques_masks.unsqueeze(-1)

        # Encoding the visual feature
        for enc in self.vis_encoder:
            vis_feat = enc(vis_feat, ques_input, ques_mask_tmp)
            vis_feat *= vis_mask.unsqueeze(-1) #vis_feat(BS, 48, 768)

        
        start_enc_output = self.prog_proj(self.embedding(program)).view(batch_size, program.size(1), -1)    # [BS, R, D] [32, 9, 768]
        # start_enc_output = self.prog_proj2(self.prog_linear(self.embedding(program))).view(batch_size, program.size(1), -1)
        
        # start_enc_output = self.prog_self_attention(start_enc_output)
        transition_masks = transition_masks.transpose(0, 1)  # [T, BS, R, R] [5, 512, 9, 9]
        activate_masks = activate_masks.transpose(0, 1)  # [T, BS, R]  [5, 512, 9]

        enc_output = start_enc_output # [B, 9, 768]

        # Build the structure into the transformer
        for trans_mask, active_mask in zip(transition_masks, activate_masks):
            last_enc_output = enc_output.clone()
            trans_mask = ((1 - trans_mask) * -1e9).unsqueeze(1)
            # import pdb; pdb.set_trace()
            enc_output = self.exec_cross_attention(enc_output, vis_feat) # [B, 9, 768]
            enc_output = self.exec_self_attention(enc_output, trans_mask) # [B, 9, 768]
            # enc_output = self.exec_cross_attention(enc_output, vis_feat) # [B, 9, 768]
            enc_output = enc_output * program_masks.unsqueeze(-1) # [B, 9, 768]
            active_mask = active_mask.unsqueeze(-1)
            enc_output = active_mask * enc_output + (1 - active_mask) * last_enc_output


        # Post-Processing the encoder output
        for layer in self.post:
            enc_output = layer(enc_output, vis_feat)

        # Predict the intermediate results
        pre_logits = self.idx_predictor(enc_output)
        lang_feat = torch.gather(enc_output, 1, index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))
        lang_feat = lang_feat.view(batch_size, -1)
        # adapted_w = torch.softmax(self.adapted_w, 0)
        if self.forward_ques:
            ques_feat = torch.mean(ques_input, dim=1) #ques_input:(BS, 30, 768)
            logits = self.proj(torch.cat([lang_feat, ques_feat], dim=-1))
        elif self.forward_vis:
            vis_feat = torch.mean(vis_feat, dim = 1)
            logits = self.proj(torch.cat([lang_feat, vis_feat], dim=-1))
        else:
            logits = self.proj(lang_feat)

        # pre_logits: [BS, P, N], logits:[BS, A]
        return pre_logits, logits


if __name__ == '__main__':
    test_m = SoftLogicTransformer(vocab_size=3762, stacking=2, answer_size=1845, visual_dim=2048,
                                  coordinate_dim=6, n_head=8, n_layers=5, dropout=0.1,
                                  intermediate_dim=49, pre_layers=3, intermediate_layer=False)
    print(test_m)
    # ques, ques_masks, prog, prog_masks, trans_masks, activate_masks, object_feat, box_feat, vis_mask, index, depth = \
    #     torch.tensor()
