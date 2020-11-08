from utils.dict import Dictionary
import torch
import os
from tqdm import tqdm
import pickle
import math


def get_preds(docs, probs, lengths):
    '''
    Args:
        docs: of shape(batch_size, seq_len)
        probs: of shape(batch_size, seq_len)
        lengths: of shape(batch_size)
    Return:
        indexs: [{word_id: prob}], length of the list = batch_size
    '''
    batch_size = lengths.shape[0]
    indexs = [{} for i in range(batch_size)]
    for i in range(batch_size):
        seq = docs[i]
        len = lengths[i]
        for j in range(len):
            if seq[j] in indexs[i].keys():
                indexs[i][seq[j]] += probs[i][j]
            else:
                indexs[i][seq[j]] = probs[i][j]

    return indexs

class myDataloader():
    '''Dataloader is a class which aims to convert sequence vector to one-hot matrix
    and bacth the data
    '''

    def __init__(self, dic: Dictionary, data, batch_size: int):
        '''Initialize the dataloader

        Args:
            dic: Dictionary(), the dictionary of the dataset.
            vec: list, of shape[{'document':doc_vec, 'query':query_vec, 'answer':answer_vec},...], in the same length of file nums
            batch_size: set as the batch size you need
        '''
        self.dic = dic
        self.batch_size = batch_size
        self.document = data['document']
        self.query = data['query']
        self.answer = data['answer']
        self.sample_num = len(self.document)
        self.batch_num = math.ceil(self.sample_num / self.batch_size)
        # files = [dir+filename for filename in os.listdir(dir)]
        # doc = []
        # query = []
        # answer = []
        # for file in tqdm(files):
        #     with open(file, 'r',  encoding='utf8') as f:
        #         whole = f.readlines()
        #         document = remove_sig(whole[2]).split()
        #         query = remove_sig(whole[4]).split()
        #         answer = remove_sig(whole[6])
        #         doc.append([self.dic.getId(word) for word in document])
        #         query.append([self.dic.getId(word) for word in query])
        #         answer.append([self.dic.getId(word) for word in answer.split()])
        # self.vector['document'] = torch.tensor(doc)
        # self.vector['query'] = torch.tensor(query)
        # self.vector['answer'] = torch.tensor(answer)

    # def _vec2onehot(self):
    #     dic_len = self.dic.len
    #     for f in self.vec:
    #         document_vec = torch.tensor([[word] for word in f['document']])
    #         query_vec = torch.tensor([[word] for word in f['query']])
    #         answer_vec = torch.tensor(f['answer'])
    #         onehot_file = {
    #             'document': torch.zeros(self.seq_len['document'], dic_len).scatter_(1, document_vec, 1),
    #             'query': torch.zeros(self.seq_len['query'], dic_len).scatter_(1, query_vec, 1),
    #             'answer': answer_vec
    #         }
    #         self.onehot.append(onehot_file)

    def shuffle(self):
        '''shuffle the dataset
        '''
        data = list(zip(self.document,
                        self.query, self.answer))
        self.document, self.query, self.answer = zip(
            *[data[i] for i in torch.randperm(len(data))])

    def _batch(self, data, output_lengths=True):
        lengths = torch.tensor([len(i) for i in data])
        max_length = max(lengths)
        out = torch.zeros(len(data), max_length)
        for i in range(len(data)):
            for j in range(lengths[i]):
                out[i][j] = data[i][j]
        if output_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.batch_num, "index %d > batch num %d" % (
            index, self.batch_num)

        docs, doc_lengths = self._batch(
            self.document[index * self.batch_size:(index+1)*self.batch_size])

        querys, query_lengths = self._batch(
            self.query[index * self.batch_size:(index+1)*self.batch_size])

        answers = torch.tensor(self.answer[index *
                                           self.batch_size:(index+1)*self.batch_size])

        return (docs.long(), doc_lengths), (querys.long(), query_lengths), answers
