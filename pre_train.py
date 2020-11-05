from tqdm import tqdm
from collections import Counter
from utils.dict import Dictionary
import pickle
import torch
from utils.dataloader import myDataloader
import re
import os

def remove_sig(str: str):
    '''remove_sig, remove signals from the input string
    Args:
        str: the input string

    Returns:
        A string without signals like .'", etc
    '''
    return re.sub("[+\.\!\/<>“”''"",$?\-%^*():+\"\']+|[+——！，。？、~#￥%……&*（）]+", "", str.strip())

def build_dict(dirs):
    '''build a dictionary for all vocabulary in the dataset
    Args:
        dirs: a list consists of the path to train, test, valid
    Returns:
        dic: Counter(), the dictionary
    '''
    print("building dictionary...")
    files = [dir+file for dir in dirs for file in os.listdir(dir)]
    dic = Counter()
    for file in tqdm(files):
        with open(file, 'r',  encoding='utf8') as f:
            whole = f.readlines()
            document = remove_sig(whole[2])
            query = remove_sig(whole[4])
            answer = remove_sig(whole[6])
            for word in document.split()+query.split()+answer.split():
                dic[word] += 1
    print("building dictionary finished!")
    return dic

def vectorize(dirs, dic, vec_cache):
    '''convert the text sequence to vector
    Args:
        dirs: path to train, test, valid dataset
        dic: Dictionary(), the dictionary
        vec_cache: the path of temporary files to be saved
    Returns:
        No Return
    '''
    print("start vectorizing...")
    files = [[dir+file for file in os.listdir(dir)] for dir in dirs]
    vec = [{},{},{}]
    for i in range(3):
        docs = []
        querys = []
        answers = []
        for file in tqdm(files[i]):
            with open(file, 'r',  encoding='utf8') as f:
                whole = f.readlines()
                document = remove_sig(whole[2]).split()
                query = remove_sig(whole[4]).split()
                answer = remove_sig(whole[6])
                docs.append([dic.getId(word) for word in document])
                querys.append([dic.getId(word) for word in query])
                answers.append([dic.getId(word) for word in answer.split()])
        vec[i]['document'] = docs
        vec[i]['query'] = querys
        vec[i]['answer'] = answers
        with open(vec_cache[i], 'wb') as f:
            pickle.dump(vec[i], f)
    print("vectorizing finished, file saved.")

def main():
    dataset_path = 'C:/Users/angus/Desktop/论文复现/code/train_input/datasets'
    dirs = ['/cnn/questions/training/', '/cnn/questions/test/', '/cnn/questions/validation/']
    dirs = [dataset_path+dir for dir in dirs]
    dic_cache = './temp/dictionary.pickle'
    if os.path.exists(dic_cache):
        print("dictionary cache file existed!")
    else:
        print("dictionary cache file not found!")
        dic = build_dict(dirs)
        sorted_dic, _ = zip(*dic.most_common())
        word2id = {token: i for i,token in enumerate(sorted_dic)}
        dictionary = Dictionary(word2id)
        with open(dic_cache, 'wb') as f:
            pickle.dump(dictionary, f)

    # 将文本转换为其id序列
    vec_cache = ['./temp/train_vec.pickle', './temp/test_vec.pickle', './temp/valid_vec.pickle']
    vec_cache_exist = True
    for i in range(3):
        if not os.path.exists(vec_cache[i]):
            vec_cache_exist = False
            for cache in vec_cache:
                if os.path.exists(cache):
                    os.remove(cache)
            break
    if vec_cache_exist:
        print("vector cache file exists!")
    else:
        print("vector cache file not found!")
        vectorize(dirs, dictionary, vec_cache)


if __name__ == '__main__':
    main()

    # 下面是在train用来加载数据的例子
    # 加载字典
    with open('./temp/dictionary.pickle', 'rb') as f:
        dictionary = pickle.load(f)

    # 加载数据
    with open('./temp/train_vec.pickle', 'rb') as f: 
        train_data = pickle.load(f)

    batched_train_data = myDataloader(dictionary, train_data, 32)
    (docs, doc_lengths), (querys, query_lengths), answers = batched_train_data[2]
    print(doc_lengths.shape)