import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weigth_init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def sort_batch(data, lens):
    sorted_lens, sorted_idx = torch.sort(lens, dim=0, descending=True)
    sorted_data = data[sorted_idx]
    _, recover_idx = torch.sort(sorted_idx, dim=0, descending=False)
    return (sorted_data, sorted_lens.cuda()), recover_idx.cuda()


def softmax_mask(input, mask, axis=1, epsilon=1e-12):
    shift, _ = torch.max(input, axis, keepdim=True)
    shift = shift.expand_as(input).cuda()
    input = torch.exp(input - shift) * mask
    sum = torch.sum(input, axis, keepdim=True).expand_as(input)
    softmax = input / (sum + epsilon)
    return softmax.cuda()


class ATT_model(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_dim,
                 dropout_rate,
                 PAD,
                 bidirectional=True):
        super(ATT_model, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(self.vocab_size+1,
                                      self.embed_dim,
                                      padding_idx=PAD)
        self.embedding.weight.data.uniform_(-0.05, 0.05)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.BiGRU = torch.nn.GRU(input_size=self.embed_dim,
                                  hidden_size=self.hidden_dim,
                                  bidirectional=bidirectional,
                                  batch_first=True)

        for weight in self.BiGRU.parameters():
            if len(weight.size()) > 1:
                weigth_init.orthogonal_(weight.data)

    def forward(self, documents, doc_lens, doc_masks, querys, query_lens, query_masks, answers):

        (sorted_documents,
         sorted_doc_lens), recover_idx_doc = sort_batch(documents, doc_lens)
        (sorted_querys, sorted_query_lens), recover_idx_query = sort_batch(
            querys, query_lens)

        doc_embedding = self.embedding(sorted_documents)
        query_embedding = self.embedding(sorted_querys)

        doc_embedding_drop = self.dropout(doc_embedding)
        query_embedding_drop = self.dropout(query_embedding)

        doc_embedding_packed = pack(doc_embedding_drop,
                             sorted_doc_lens,
                             batch_first=True)
        query_embedding_packed = pack(query_embedding_drop,
                               sorted_query_lens,
                               batch_first=True)

        # GRU
        document_gru_output, _ = self.BiGRU(doc_embedding_packed, None)
        query_gru_output, _ = self.BiGRU(query_embedding_packed, None)

        document_gru_output, _ = unpack(document_gru_output, batch_first=True)
        query_gru_output, _ = unpack(query_gru_output, batch_first=True)
        document_gru_output = document_gru_output[recover_idx_doc]
        query_gru_output = query_gru_output[recover_idx_query]


        # Pair-wise Matching score
        IndiATT_output = torch.bmm(document_gru_output,
                                   query_gru_output.transpose(2, 1))
        IndiATT_mask = torch.bmm(doc_masks, query_masks.transpose(2, 1))

        # ATT-over-ATT
        alpha = softmax_mask(IndiATT_output, IndiATT_mask, axis=1)  # 列
        beta = softmax_mask(IndiATT_output, IndiATT_mask, axis=2)  # 行

        beta_sum = torch.sum(beta, dim=1, keepdim=True)
        beta_aver = beta_sum / doc_lens.unsqueeze(1).unsqueeze(2).expand_as(
            beta_sum)
        AoA_output = torch.bmm(alpha, beta_aver.transpose(2, 1))

        answers_expand = answers.expand_as(documents)
        answer_masks = answers_expand == documents

        # probs 表示真实答案在预测中的概率
        probs = []

        for i in range(answers.shape[0]):
            probs.append(
                torch.sum(torch.masked_select(AoA_output[i].squeeze(), answer_masks[i]), 0, keepdim=True)
            )
        probs = torch.cat(probs, 0).squeeze()

        # pred_answers 表示从documents中选取概率最大的单词作为预测
        pred_answers = []
        for j, document in enumerate(documents):
            dict = {}
            len = doc_lens[j]
            s = AoA_output[j].squeeze()
            for i, word in enumerate(document[:len]):
                if word not in dict:
                    dict[word] = s[i].data
                else:
                    dict[word] += s[i].data
            pred_answers.append(max(dict, key=dict.get))
        pred_answers = torch.LongTensor(pred_answers).cuda()
#         debug = []
#         for i, document in enumerate(documents):
#             prob = []
#             s = AoA_output[i].squeeze()
#             for j, word in enumerate(document[:doc_lens[i]]):
#                 mask = document == word.expand_as(document)
#                 prob.append(torch.sum(torch.masked_select(s, mask), 0, keepdim=True))
#             prob = torch.cat(prob, dim=0).squeeze()
#             _ , max_index = torch.max(prob, 0)
#             pred_answers.append(document.index_select(0, max_index))
#             debug.append(prob)
#         pred_answers = torch.cat(pred_answers, dim=0).squeeze()

        return probs, pred_answers
