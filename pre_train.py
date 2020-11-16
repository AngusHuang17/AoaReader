from tqdm import tqdm
from collections import Counter
from utils.dict import Dictionary
import pickle
import re
import os


def remove_sig(str: str):
    '''remove_sig, remove signals from the input string
    Args:
        str: the input string

    Returns:
        A string without signals like .'", etc
    '''
    return re.sub("[+\.\!\/<>“”''"
                  ",$?\-%^*():+\"\']+|[+——！，。？、~#￥%……&*（）]+", "", str.strip())


def build_dict(dirs):
    '''build a dictionary for all vocabulary in the dataset
    Args:
        dirs: a list consists of the path to train, test, valid
    Returns:
        dic: Counter(), the dictionary
    '''
    print("building dictionary...")
    files = [dir + file for dir in dirs for file in os.listdir(dir)]
    dic = Counter()
    for file in tqdm(files):
        with open(file, 'r', encoding='utf8') as f:
            whole = f.readlines()
            document = whole[2].lower()
            query = whole[4].lower()
            answer = whole[6].lower()
            for word in document.split() + query.split() + answer.split():
                dic[word] += 1
    print("building dictionary finished!")
    return dic


def vectorize(dirs, dic):
    '''convert the text sequence to vector
    Args:
        dirs: path to train, test, valid dataset
        dic: Dictionary(), the dictionary
        vec_cache: the path of temporary files to be saved
    Returns:
        No Return
    '''
    print("start vectorizing...")
    files = [[dir + file for file in os.listdir(dir)] for dir in dirs]
    vec = [{} for i in range(len(dirs))]
    for i in range(len(dirs)):
        docs = []
        querys = []
        answers = []
        for file in tqdm(files[i]):
            with open(file, 'r', encoding='utf8') as f:
                whole = f.readlines()
                document = whole[2].lower().split()
                query = whole[4].lower().split()
                answer = whole[6].lower().split()
                docs.append([dic.getId(word) for word in document])
                querys.append([dic.getId(word) for word in query])
                answers.append([dic.getId(word) for word in answer])
        vec[i]['document'] = docs
        vec[i]['query'] = querys
        vec[i]['answer'] = answers
        # with open(vec_cache[i], 'wb') as f:
        #     pickle.dump(vec[i], f)
    print("vectorizing finished, file saved.")
    return vec

'''
def main():
    print('请输入cnn数据集文件夹的路径（path）：注意，目录结构为 path/cnn/questions, path/cnn/valid, path/cnn/test')
    dataset_path = input()
    dirs = [
        '/cnn/questions/training/', '/cnn/questions/test/',
        '/cnn/questions/validation/'
    ]
    dirs = [dataset_path + dir for dir in dirs]
    dic_cache = './temp/dictionary.pickle'
    if os.path.exists(dic_cache):
        print("dictionary cache file existed!")
        with open(dic_cache, 'rb') as f:
            dictionary = pickle.load(f)
    else:
        print("dictionary cache file not found!")
        dic = build_dict(dirs)
        sorted_dic, _ = zip(*dic.most_common())
        word2id = {token: i + 1 for i, token in enumerate(sorted_dic)}
        dictionary = Dictionary(word2id)
        with open(dic_cache, 'wb') as f:
            pickle.dump(dictionary, f)

    # 将文本转换为其id序列
    print('Vocab size:', dictionary.len)
    
    vec_cache = [
        './temp/train_vec.pickle', './temp/test_vec.pickle',
        './temp/valid_vec.pickle'
    ]
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

    # 下面是在train用来加载数据的例子，实际使用时可以注释掉
    # 加载字典
    # with open('./temp/dictionary.pickle', 'rb') as f:
    #     dictionary = pickle.load(f)

    # # 加载数据
    # with open('./temp/train_vec.pickle', 'rb') as f:
    #     train_data = pickle.load(f)

    # batched_train_data = myDataloader(dictionary, train_data, 32)
    # (docs, doc_lengths), (querys,
    #                       query_lengths), answers = batched_train_data[2]
    # print(doc_lengths.shape)
'''