import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def softmax_mask(input, mask, axis=1):
    input = torch.exp(input) * mask
    sum = torch.sum(input, axis, keepdim=True).expand_as(input)
    softmax = input/sum
    return softmax

class ATT_model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout_rate, bidirectional=True, PAD=0):
        super(AOA_model, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=PAD)
        #dropout?
        self.BiGRU = torch.nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim, dropout=self.dropout_rate, bidirectional=bidirectional,batch_first=True)
        
    def forword(self, document, query, doc_mask, query_mask):#document: 32*11255*384, ndarray, query: 32*260*384, ndarray
 
        # doc_aligned = 
        # query_aligned = 

        #max_len是所有文档的最长还是当前batch的最长
        doc_embedding = pack(self.embedding(doc_aligned), self.doc_max_len, batch_first=True)
        query_embedding = pack(self.embedding(query_aligned), self.query_max_len, batch_first=True)


        #GRU
        document_gru_output, _ = self.BiGRU_query(doc_embedding, None) 
        query_gru_output, _ = self.BiGRU_query(query_embedding, None) 

        document_gru_output, _ = unpack(document_gru_output, batch_first=True)
        query_gru_output, _ = self.unpack(query_gru_output, batch_first=True)

        #Individual ATT
        IndiATT_output = torch.bmm(document_gru_output, query_gru_output.transpose(2,1))
        IndiATT_mask = torch.bmm(doc_mask, query_mask.transpose(2,1))

        #ATT-over-ATT
        alpha = softmax_mask(IndiATT_output, IndiATT_mask, axis=1)#列
        beta = softmax_mask(IndiATT_output, IndiATT_mask, axis=2)#行
        
        beta_aver = torch.mean(beta, 1, keepdim=True)
        AoA_output = torch.bmm(alpha, beta_aver.transpose(2,1))

        word_prob = (AoA_output)

        return word_prob

        



        



