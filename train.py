import torch
import argparse
import pickle
from utils.dict import Dictionary
from utils.dataloader import myDataloader
from model.model import ATT_model
from torch.optim import Adam

'''
step1: 数据读入
step2: 实例化模型
step3: 使用一个for循环进行训练，其步骤有模型读入数据输出，计算损失函数，反向传播，优化算法更新参数
step4: 存储模型
'''

# 命令行参数设定
parser = argparse.ArgumentParser(description='Train AoAReader model.')

parser.add_argument(
    '-traindata',
    default='./temp/train_vec.pickle',
    help=
    'Path to the train_vec.pickle file from pre_train.py, default value is \'./temp/train_vec.pickle\''
)
parser.add_argument(
    '-validdata',
    default='./temp/valid_vec.pickle',
    help=
    'Path to the valid_vec.pickle file from pre_train.py, default value is \'./temp/valid_vec.pickle\''
)
parser.add_argument(
    '-dict',
    default='./temp/dictionary.pickle',
    help=
    'Path to the dictionary file from pre_train.py, default value is \'./temp/dictionary.pickle\''
)
parser.add_argument('-model_path',
                    default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_ACC.pt to 'models/' directory, where ACC is the
                    validation accuracy""")

parser.add_argument('-batch_size', type=int, default=32, help='batch size')
parser.add_argument('-embedding_size',
                    type=int,
                    default=384,
                    help='Embedding layer size')
parser.add_argument('-gru_size', type=int, default=256, help='GRU layer size')
parser.add_argument('-epoch',
                    type=int,
                    default=10,
                    help='Number of training epochs')
parser.add_argument('-lr',
                    type=float,
                    default=0.001,
                    help='Learning rate of optimizer')
parser.add_argument('-l2',
                    type=float,
                    default=0.0001,
                    help='L2 regularization')
parser.add_argument('-dropout', type=float, default=0.1, help='Dropout rate')

params = parser.parse_args()
print(params)


def loss_func(true_answers, pred_answers, probs):
    '''Calculate the loss with formulate loss = -sum(log(p(x))), x in answers
    
    Args:
        true_answers: the answers of a batch
        pred_answers: (tensor(batch_size)) predicted answers of a batch
        probs: (tensor(batch_size)) probability of true answer in predict vector s
    Returns:
        loss: -sum(log(probs(x)))
        correct_num: numbers of (true_answer==pred_answer)
    '''
    loss = -1 * torch.sum(torch.log(probs))
    compare = true_answers.squeeze() == pred_answers
    correct_num = 0
    for i in compare:
        if i:
            correct_num += 1
    return loss, correct_num


# TODO: 1.是否设置检查点； 2.完善valid的评估
def trainModel(model, train_data, valid_data, optimizer):
    def trainEpoch(epoch):
        train_data.shuffle()
        batch_num = train_data.batch_num

        total_correct = 0
        total_loss = 0
        total = 0
        for i in range(batch_num):
            (docs, doc_lengths), (querys,
                                  query_lengths), answers = train_data[i]
            optimizer.zero_grad()

            probs, pred_answers = model(docs, doc_lengths, querys,
                                        query_lengths, answers)

            loss, pred_correct = loss_func(answers, pred_answers, probs)

            loss.backward()

            # set gradient clipping threshold to 5
            for parameter in model.parameters():
                parameter.grad.data.clamp_(-5.0, 5.0)

            optimizer.step()

            total_loss += loss
            total_correct += pred_correct
            total += answers.shape[0]

            print(
                'Epoch %d, %d th batch, avg loss: %.2f, avg correct_num: %.2f'
                % (epoch, i, loss / answers.shape[0], pred_correct / answers.shape[0]))

            del loss, pred_answers, probs

        return total_loss / total, total_correct / total


    for epoch in range(params.epoch):
        train_loss, train_acc = trainEpoch(epoch)
        print('Epoch %d:\t average loss: %.2f\t train accuracy: %g' %
              (epoch, train_loss, train_acc * 100))
         
        # model_state_dict = model.state_dict()
        # optimizer_state_dict = optimizer.state_dict()

        # checkpoint = {
        #     'model': model_state_dict,
        #     'epoch': epoch,
        #     'optimizer': optimizer_state_dict,
        #     'opt': params,
        # }
        # torch.save(
        #     checkpoint, 'model/%s_epoch%d_acc_%.2f.pt' %
        #     (opt.save_model, epoch, 100 * valid_acc))


# TODO: Score函数，score随着epoch绘制图变化曲线



# TODO: 目前已经可以在CPU上面进行训练，问题是训练速度太慢，考虑使用cuda在gpu训练，而后部署到华为云上
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

    # 模型实例化
    model = ATT_model(vocab_size=dictionary.len,
                      embed_dim=params.embedding_size,
                      hidden_dim=params.gru_size,
                      dropout_rate=params.dropout,
                      PAD=0)

    # 优化器
    optimizer = Adam(model.parameters(),
                     lr=params.lr,
                     weight_decay=params.dropout)

    # 训练模型
    trainModel(model, batched_train_data, batched_valid_data,optimizer)


if __name__ == "__main__":
    main()
