import torch, math, itertools, os, psutil
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from itertools import permutations, product

from torch.nn.init import xavier_normal_, xavier_uniform_, uniform_, zeros_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class RETA(torch.nn.Module):

    def __init__(self, num_relations, num_entities, num_types, embedding_size, num_filters):
        super(RETA, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters

        self.f_FCN_net = torch.nn.Linear((num_filters*(embedding_size-2))*2, 1)
        xavier_normal_(self.f_FCN_net.weight.data)
        zeros_(self.f_FCN_net.bias.data)

        self.emb_relations = torch.nn.Embedding(num_relations, self.embedding_size, padding_idx=0)
        self.emb_entities = torch.nn.Embedding(num_entities, self.embedding_size, padding_idx=0)
        self.emb_types = torch.nn.Embedding(num_types, self.embedding_size, padding_idx=0)

        self.conv1 = torch.nn.Conv2d(1, num_filters, (3, 3))
        zeros_(self.conv1.bias.data)
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1)

        self.conv2 = torch.nn.Conv2d(1, num_filters, (3, 3))
        zeros_(self.conv2.bias.data)
        self.batchNorm2 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv2.weight, mean=0.0, std=0.1)

        self.loss = torch.nn.Softplus()

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_relations.weight.data, -bound, bound)
        uniform_(self.emb_entities.weight.data, -bound, bound)
        uniform_(self.emb_types.weight.data, -bound, bound)

    def forward(self, x_batch, arity, mode, device=None, id2relation=None, id2entity=None):

        # H-R-T
        fact_relations_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,0:2]).flatten()).cuda(device)
        fact_entities_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,0:2]).flatten()).cuda(device)
        fact_relations_embedded = self.emb_relations(fact_relations_ids).view(len(x_batch),2,self.embedding_size)
        fact_entities_embedded = self.emb_entities(fact_entities_ids).view(len(x_batch),2,self.embedding_size)

        fact_hrt_concat1 = torch.cat((fact_entities_embedded[:,0,:].unsqueeze(1), fact_relations_embedded[:,0,:].unsqueeze(1), fact_entities_embedded[:,1,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors1 = self.conv1(fact_hrt_concat1)
        fact_hrt_concat_vectors1 = self.batchNorm1(fact_hrt_concat_vectors1)
        fact_hrt_concat_vectors1 = F.relu(fact_hrt_concat_vectors1).squeeze(3)
        fact_hrt_concat_vectors1 = fact_hrt_concat_vectors1.view(fact_hrt_concat_vectors1.size(0), -1).unsqueeze(2)

        # TYPES
        head_types_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,2:]).flatten()).cuda(device)
        tail_types_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,2:]).flatten()).cuda(device)
        head_types_embedded = self.emb_types(head_types_ids).view(len(x_batch),arity-2,self.embedding_size)
        tail_types_embedded = self.emb_types(tail_types_ids).view(len(x_batch),arity-2,self.embedding_size)

        headType_relation_tailType_concat = torch.cat((head_types_embedded[:,0,:].unsqueeze(1), fact_relations_embedded[:,0,:].unsqueeze(1), tail_types_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
        headType_relation_tailType_concat = self.conv2(headType_relation_tailType_concat)
        headType_relation_tailType_concat = self.batchNorm2(headType_relation_tailType_concat)
        headType_relation_tailType_concat = F.relu(headType_relation_tailType_concat).squeeze(3)
        headType_relation_tailType_concat = headType_relation_tailType_concat.view(headType_relation_tailType_concat.size(0), -1).unsqueeze(2)

        for i in range(arity-3):
            headType_relation_tailType_concat_tmp = torch.cat((head_types_embedded[:,i+1,:].unsqueeze(1), fact_relations_embedded[:,0,:].unsqueeze(1), tail_types_embedded[:,i+1,:].unsqueeze(1)), 1).unsqueeze(1)
            headType_relation_tailType_concat_tmp = self.conv2(headType_relation_tailType_concat_tmp)
            headType_relation_tailType_concat_tmp = self.batchNorm2(headType_relation_tailType_concat_tmp)
            headType_relation_tailType_concat_tmp = F.relu(headType_relation_tailType_concat_tmp).squeeze(3)
            headType_relation_tailType_concat_tmp = headType_relation_tailType_concat_tmp.view(headType_relation_tailType_concat_tmp.size(0), -1).unsqueeze(2)
            headType_relation_tailType_concat = torch.cat((headType_relation_tailType_concat, headType_relation_tailType_concat_tmp), 2)

        min_val, _ = torch.min(headType_relation_tailType_concat, 2)

        # COMBINING H-R-T and TYPES
        fact_hrt_concat_vectors1 = fact_hrt_concat_vectors1.squeeze(2)
        concat_hrt_and_type = torch.cat((fact_hrt_concat_vectors1, min_val), 1)
        evaluation_score = self.f_FCN_net(concat_hrt_and_type)

        return evaluation_score

class RETA_NO_TYPES(torch.nn.Module):

    def __init__(self, num_relations, num_entities, num_types, embedding_size, num_filters):
        super(RETA_NO_TYPES, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters

        self.f_FCN_net = torch.nn.Linear((num_filters*(embedding_size-2)), 1)
        xavier_normal_(self.f_FCN_net.weight.data)
        zeros_(self.f_FCN_net.bias.data)

        self.emb_relations = torch.nn.Embedding(num_relations, self.embedding_size, padding_idx=0)
        self.emb_entities = torch.nn.Embedding(num_entities, self.embedding_size, padding_idx=0)
        self.emb_types = torch.nn.Embedding(num_types, self.embedding_size, padding_idx=0)

        self.conv1 = torch.nn.Conv2d(1, num_filters, (3, 3))
        zeros_(self.conv1.bias.data)
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1)

        self.loss = torch.nn.Softplus()

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_relations.weight.data, -bound, bound)
        uniform_(self.emb_entities.weight.data, -bound, bound)
        uniform_(self.emb_types.weight.data, -bound, bound)

    def forward(self, x_batch, arity, mode, device=None, id2relation=None, id2entity=None):

        # H-R-T
        fact_relations_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,0:2]).flatten()).cuda(device)
        fact_entities_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,0:2]).flatten()).cuda(device)
        fact_relations_embedded = self.emb_relations(fact_relations_ids).view(len(x_batch),2,self.embedding_size)
        fact_entities_embedded = self.emb_entities(fact_entities_ids).view(len(x_batch),2,self.embedding_size)

        fact_hrt_concat1 = torch.cat((fact_entities_embedded[:,0,:].unsqueeze(1), fact_relations_embedded[:,0,:].unsqueeze(1), fact_entities_embedded[:,1,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors1 = self.conv1(fact_hrt_concat1)
        fact_hrt_concat_vectors1 = self.batchNorm1(fact_hrt_concat_vectors1)
        fact_hrt_concat_vectors1 = F.relu(fact_hrt_concat_vectors1).squeeze(3)
        fact_hrt_concat_vectors1 = fact_hrt_concat_vectors1.view(fact_hrt_concat_vectors1.size(0), -1).unsqueeze(2)

        fact_hrt_concat_vectors1 = fact_hrt_concat_vectors1.squeeze(2)
        evaluation_score = self.f_FCN_net(fact_hrt_concat_vectors1) 

        return evaluation_score
