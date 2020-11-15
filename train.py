import torch
import argparse
import pickle
from utils.dict import Dictionary
from utils.dataloader import myDataloader
from model.model import ATT_model
from torch.optim import Adam
import time



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


parser.add_argument('-batch_size', type=int, default=32, help='batch size')

parser.add_argument('-embedding_size',
                    type=int,
                    default=384,
                    help='Embedding layer size')
parser.add_argument('-gru_size', type=int, default=256, help='GRU layer size')
parser.add_argument('-epoch',
                    type=int,
                    default=5,
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
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')


parser.add_argument('-train_from', default='', type=str,
                help="""If training from a checkpoint then this is the
                path to the pre-trained model.""")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval (minibatches).")

parser.add_argument('-model_path',
                    default='./model',
                    help="""Model filename (the model will be saved as
                    model_epochN_ACC.pt to <model_path> directory, where ACC is the
                    validation accuracy""")

parser.add_argument('-log_path', default='./', help="Path to save log file.")


parser.add_argument('-gpu', default=True, type=bool,
                    help="whether to use gpu")

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
    loss = - torch.mean(torch.log(probs), dim=0, keepdim=True)
    num_correct = (true_answers.squeeze() == pred_answers).sum().squeeze().data
    return loss.cuda(), num_correct.float()

def eval(model, data):
    total_correct = 0
    total_loss = 0
    total_sample_num = 0

    batch_num = data.batch_num
    
    model.eval()
    for i in range(batch_num):
        (docs, doc_lengths, doc_masks), (querys, query_lengths, query_masks), answers = data[i]
        probs, pred_answers = model(docs, doc_lengths, doc_masks,
                                    querys, query_lengths, query_masks,
                                    answers)
        loss, pred_correct = loss_func(answers, pred_answers, probs)

        total_loss += loss.data[0]
        total_correct += pred_correct.data
        total_sample_num += answers.shape[0]

        del loss, pred_answers, probs

    model.train()
    return total_loss / total_sample_num, total_correct / total_sample_num



def trainModel(model, train_data, valid_data, optimizer):

    start_time = time.time()

    def trainEpoch(epoch):
        train_data.shuffle()
        batch_num = train_data.batch_num

        total_correct = 0
        total_loss = 0
        total_sample_num = 0

        for i in range(batch_num):
            (docs, doc_lengths, doc_masks), (querys, query_lengths, query_masks), answers = train_data[i]

            optimizer.zero_grad()

            probs, pred_answers = model(docs, doc_lengths, doc_masks,
                                        querys, query_lengths, query_masks,
                                        answers)

            loss, pred_correct = loss_func(answers, pred_answers, probs)

            loss.backward()

            # set gradient clipping threshold to 5
            for parameter in model.parameters():
                parameter.grad.data.clamp_(-5.0, 5.0)

            # update parameters
            optimizer.step()

            current_batch_num = answers.shape[0]
            total_loss += loss.data[0] * current_batch_num
            total_correct += pred_correct.data
            total_sample_num += current_batch_num

            end_time = time.time()

            if i % params.log_interval == 0:
                print( "Epoch %d, %d th batch, avg loss: %.2f, acc: %6.2f; %6.0f s elapsed"
                    % (epoch, i, total_loss / total_sample_num, total_correct / total_sample_num * 100,
                       end_time - start_time))

            del loss, pred_answers, probs

        return total_loss / total_sample_num, total_correct / total_sample_num

    for epoch in range(params.epoch):

        # 1. 训练集上训练一个epoch
        train_loss, train_acc = trainEpoch(epoch)

        print('Epoch %d:\t average loss: %.2f\t train accuracy: %g' % (epoch, train_loss, train_acc * 100))

        # 2. 验证集上评估
        valid_loss, valid_acc = eval(model, valid_data)
        print('=' * 20)
        print('Evaluating on validation set:')
        print('Validation loss: %.2f' % valid_loss)
        print('Validation accuracy: %g' % (valid_acc * 100))
        print('=' * 20)

        # 3. 每个epoch保存模型
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'epoch': epoch,
            'optimizer': optimizer_state_dict,
            'params': params,
        }
        torch.save( checkpoint, params.model_path + '/model_epoch%d_acc_%.2f.pt' % (epoch, 100 * valid_acc))


def main():
    # 打印当前时间
    print('-' * 20)
    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)
    print('-' * 20)
    
    # 清空显存
    torch.cuda.empty_cache()
    
    global params
    train_from = params.train_from
    if params.train_from:
        train_from = True
        print("Trains from " + params.train_from)
        checkpoint = torch.load(params.train_from)
        params = checkpoint['params']
    
    # 加载字典
    with open(params.dict, 'rb') as f:
        dictionary = pickle.load(f)

    # 加载数据
    with open(params.traindata, 'rb') as t, open(params.validdata, 'rb') as v:
        train_vec = pickle.load(t)
        valid_vec = pickle.load(v)

    batched_train_data = myDataloader(dictionary, train_vec, True, params.batch_size)
    batched_valid_data = myDataloader(dictionary, valid_vec, True, params.batch_size)

    # 模型实例化
    model = ATT_model(vocab_size=dictionary.len,
                      embed_dim=params.embedding_size,
                      hidden_dim=params.gru_size,
                      dropout_rate=params.dropout,
                      PAD=0)
    # 模型参数个数
    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
                         
    # 使用多个gpu加速
    if params.gpu:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
                         
    
    if train_from:
        print('Loading model from checkpoint at %s' % params.train_from)
        chk_model = checkpoint['model']
        model.load_state_dict(chk_model)
        params.start_epoch = checkpoint['epoch'] + 1

    # 优化器
    optimizer = Adam([{'params': model.embedding.parameters(), 'weight_decay':params.l2},
                  {'params': model.BiGRU.parameters()}], lr=params.lr)
                         
    if train_from:
       optimizer.load_state_dict(checkpoint['optimizer'])
                         


    # 训练模型并保存
    trainModel(model, batched_train_data, batched_valid_data, optimizer)


if __name__ == "__main__":
    main()
