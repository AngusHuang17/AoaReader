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

if __name__ = '__main__':
    main()