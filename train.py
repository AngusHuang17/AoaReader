<<<<<<< HEAD
<<<<<<< HEAD
import argparse
import pickle
from utils.dict import Dictionary
from utils.dataloader import myDataloader
from model.model import ATT_model

'''
step1: 数据读入
step2: 实例化模型
step3: 使用一个for循环进行训练，其步骤有模型读入数据输出，计算损失函数，反向传播，优化算法更新参数
step4: 存储模型
'''

# 命令行参数设定
parser = argparse.ArgumentParser(description='Train AoAReader model.')

parser.add_argument('-traindata', default='./temp/train_vec.pickle',
                    help='Path to the train_vec.pickle file from pre_train.py, default value is \'./temp/train_vec.pickle\'')
parser.add_argument('-validdata', default='./temp/valid_vec.pickle',
                    help='Path to the valid_vec.pickle file from pre_train.py, default value is \'./temp/valid_vec.pickle\'')
parser.add_argument('-dict', default='./temp/dictionary.pickle',
                    help='Path to the dictionary file from pre_train.py, default value is \'./temp/dictionary.pickle\'')
parser.add_argument('-model_path', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_ACC.pt to 'models/' directory, where ACC is the
                    validation accuracy""")

parser.add_argument('-batch_size', type=int, default=32, help='batch size')
parser.add_argument('-embedding_size', type=int,
                    default=384, help='Embedding layer size')
parser.add_argument('-gru_size', type=int, default=256, help='GRU layer size')
parser.add_argument('-epoch', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('-lr', type=float, default=0.001, help='Learning rate of optimizer')
parser.add_argument('-l2', type=float, default=0.0001, help='L2 regularization')
parser.add_argument('-dropout', type=float, default=0.1, help='Dropout rate')




params = parser.parse_args()
print(params)


# TODO: Score函数， loss func函数，score随着epoch图变化曲线


def main():

    # 加载字典
    with open(params.dict, 'rb') as f:
        dictionary = pickle.load(f)

    # 加载数据
    with open(params.traindata, 'rb') as tr, open(params.validdata, 'rb') as v:
        train_vec = pickle.load(tr)
        valid_vec = pickle.load(v)

    batched_train_data = myDataloader(dictionary, train_vec, params.batch_size)
    batched_valid_data = myDataloader(dictionary, valid_vec, params.batch_size)
    # (docs, doc_lengths), (querys, query_lengths), answers = batched_train_data[2]
    # print(doc_lengths.shape)

    # 模型实例化
    model = ATT_model(vocab_size=dictionary.len, embed_dim=params.embedding_size, hidden_dim=params.gru_size, dropout_rate=params.dropout, PAD=0)
=======
import torch
import math
from utils.dataloader import myDataloader
import model as model
import argparse

# torch.backends.cudnn.enabled=True

parser = argparse.ArgumentParser(description="train.py")

paser.add_argument('-batch_size', type=int, default=32)

parser.add_argument('-dropout_rate', default=0.1)

parser.add_argument('-epochs', type=int, default=13)

parser.add_argument('-start_epoch', type=int, default=1)

parser.add_argument('-learning_rate', type=int, default=0.001)

parser.add_argument('-weight_decay', type=int, default=0.0001)

parser.add_argument('-embed_dim', type=int, default=384)

parser.add_argument('-hidden_dim', type=int, default=256)

parser.add_argument('-save_model', default='model')

opt = parser.parse_args()

def loss_func(prob, answer):
    loss = torch.tensor(len(prob))
    correct = 0
    for (i, batch) in enumerate(prob):
        for word in batch:
            loss[i] = loss[i] + math.log(batch[word])
            for answerword in answer[i]:
                if word == answerword:
                    correct += 1
    return loss, correct

def trainModel(model, batched_train_data, optimizer):

    def trainEpoch(epoch):
        batched_train_data.shuffle()

        total_loss, total_correct = 0, 0
        total = len(batched_train_data)
        for i in range(len(batched_train_data)):
            (docs, doc_lengths, doc_mask), (querys, query_lengths, query_mask), answers = batched_train_data[i]
            model.zero_grad()

            probs = model(docs, doc_lengths, querys, query_lengths, doc_mask, query_mask)

            loss, pred_correct = loss_func(probs, answer)
            loss.backward()

            optimizer.step()

            total_pred_words = answers.size(0)

            total_loss += torch.sum(loss).data[0]

            total_correct += pred_correct

            del loss, probs

        return total_loss / total, total_correct / total


    for epoch in range(opt.start_epoch, opt.epoch+1):
        train_loss, train_acc = trainEpoch(epoch)
        print('Epoch %d:\t average loss: %.2f\t train accuracy: %g' % (epoch, train_loss, train_acc*100))

        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

        checkpoint = {
            'model': model_state_dict,
            'epoch': epoch,
            'optimizer': optimizer_state_dict,
            'opt': opt,
        }
        torch.save(checkpoint,
                   'model/%s_epoch%d_acc_%.2f.pt' % (opt.save_model, epoch, 100*valid_acc))


def main():
    with open('./temp/dictionary.pickle', 'rb') as f:
        dictionary = pickle.load(f)

    # 加载数据
    with open('./temp/train_vec.pickle', 'rb') as f: 
        train_data = pickle.load(f)

    batched_train_data = myDataloader(dictionary, train_data, opt.batch_size)
    model = model.model.ATT_model(embed_dim=opt.embed_dim, hidden_dim=opt.hidden_dim, dropout_rate=opt.dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    trainModel(model, batched_train_data, optimizer)

    # TODO: train the model
