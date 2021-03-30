# -*- coding: UTF-8 -*-

import networkx as nx
import sys
import os
import time
import random
import argparse
sys.path.append("../")
# import temporal_walk
from code_fast_nondecreasing import temporal_walk
from word2vec.word2vec import Word2Vec

# 没有初始化、非递减
# cd F:\dynamic_he\code\code_fast_nondecreasing
# python main.py --input-dir ../../dataset/slide_data_for_link_prediction/TKY190_full/ --output-dir ../../output/nonDecreasing_noInit/TKY190_beta999/ --node-types-file ../../dataset/TKY/TKY_node_types.txt --beta 0.999
# python main.py --input-dir ../../dataset/slide_data_for_link_prediction/enron190_skip5_full/ --output-dir ../../output/nonDecreasing_noInit/enron190_skip5_heTypeWrong/beta99/ --node-types-file ../../dataset/sig_enron/wrong_node_types_enron_dele_na_others.txt --beta 0.99
# python main.py --input-dir ../../dataset/slide_data_for_link_prediction/Tmall190_skip50_full/ --output-dir ../../output/nonDecreasing_noInit/Tmall190_skip50/ --slide-window-size 1000000000 --node-types-file ../../dataset/Tmall/Tmall_node_types.txt

# python main.py --input-dir ../../dataset/slide_data_for_link_prediction/enron190_skip5_full/ --output-dir ../../output/nonDecreasing_noInit/enron190_skip5/beta0/ --node-types-file ../../dataset/sig_enron/node_types_enron_dele_na_others.txt --beta 0
# python main.py --input-dir ../../dataset/slide_data_for_link_prediction/enron190_skip5_full/ --output-dir ../../output/nonDecreasing_noInit/enron190_skip5/beta5/ --node-types-file ../../dataset/sig_enron/node_types_enron_dele_na_others.txt --beta 0.5
# python main.py --input-dir ../../dataset/slide_data_for_link_prediction/enron190_skip5_full/ --output-dir ../../output/nonDecreasing_noInit/enron190_skip5/beta_1/ --node-types-file ../../dataset/sig_enron/node_types_enron_dele_na_others.txt --beta 1


# 已经改成快速读滑动窗口的方式了
def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Random Walk")
    # parser.add_argument('--input', nargs='?', default='../dataset/test/phone.txt',
    #                     help='Input graph path')
    parser.add_argument('--input-dir', nargs='?', default='../dataset/slide_data_for_link_prediction/TKY190_full/',
                        help='Input dir of graph')
    parser.add_argument('--output-dir', nargs='?', default='../output/test/',
                        help='Output file path')

    parser.add_argument('--node-types-file', nargs='?', default='../dataset/test/phone_node_types.txt',
                        help='Input heterogeneous node types path for JUST')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.2,
                        help='Stay probability in homogeneous for JUST')
    parser.add_argument('--beta', dest='beta', type=float, default=0.5,
                        help='Stay probability in same time for time constraint')

    parser.add_argument('--p', dest='p', type=float, default=0.2)
    parser.add_argument('--subw', dest='subw', default=2)
    # 把滑动窗口要用的初始变量替换成滑动窗口的时间长度
    # parser.add_argument('--slide-window-size', type=float, default=1000000000)
    # parser.add_argument('--n-skip', type=int, default=10000)

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    return parser.parse_args()


def train_add(trained_list, rm):
    walks = []
    for i in trained_list:
        walks += [list(map(str, walk)) for walk in rm.rm_list[i].values()]
    # 在加入word2vec模型时，先对输入的walk集合进行shuffle，可能能够避免模型过拟合
    random.shuffle(walks)

    if len(walks) > 0:
        model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers)
        model.init_sims(replace=True)
        vec = {word: model.wv.syn0[model.wv.vocab[word].index] for word in model.wv.vocab}
        return vec
    else:
        return {}


def main(args, rm, vec, input_file, outputf_rm, outputf_emb, is_init=False):
    # 进行图的构建，图传进游走部分的参数
    # g_new = nx.Graph(name='g_new')
    # load_graph(os.path.join(args.input_dir, input_file), g_new)
    graph_file = os.path.join(args.input_dir, input_file)

    rm_start = time.process_time()
    print('rm_start', rm_start)

    if is_init:
        # todo:启动时temporal random walk（有无必要再进行初始游走？）
        rm.init_walk(graph_file, outputf_rm)
    else:
        rm.back_walk(graph_file)
    rm_end = time.process_time()
    print('rm_end', rm_end)
    print(input_file[:-4], '随机游走时间(包括读图)', rm_end - rm_start)

    # 更新部分的游走序列输入dynamic skip-gram中
    sg_start = time.process_time()
    print('sg_start', sg_start)
    if is_init:
        update_vec = train_add(rm.rm_list, rm)
    else:
        update_vec = train_add(rm.change_n_list, rm)
    sg_end = time.process_time()
    print('sg_end', sg_end)
    print(input_file[:-4], '动态skipgram时间', sg_end - sg_start)
    print(input_file[:-4], '时间', sg_end - rm_start)

    # vec列表每次更新
    for node in update_vec:
        vec[node] = update_vec[node]
    print("update vec\n")

    # 生成的序列重新写文件
    with open(outputf_rm, 'w') as tf:
        for i in rm.rm_list.values():
            for j in i.values():
                if j.__len__() == 0:
                    continue
                tf.write(" ".join(map(str, j)) + "\n")
    # 更新的向量写文件
    with open(outputf_emb, "w") as ef:
        for k in update_vec:
            ef.write(k + " " + " ".join(map(str, update_vec[k])) + "\n")


if __name__ == '__main__':
    args = parse_args()
    startt = time.process_time()

    # assign output path
    output_dir_rm = args.output_dir+'rm/'
    output_dir_emb = args.output_dir+'emb/'
    if not os.path.exists(output_dir_rm):
        os.makedirs(output_dir_rm)
    if not os.path.exists(output_dir_emb):
        os.makedirs(output_dir_emb)

    files = os.listdir(args.input_dir)
    # files.sort()  # 按字符串排序
    files.sort(key=lambda x: int(x[:-4]))  # 文件名按整数排序
    print('num of files:', os.listdir(args.input_dir).__len__())
    print(files)

    # 游走(游走结果写文件)
    rm = temporal_walk.JustRandomWalkGenerator(args.node_types_file, args.num_walks, args.walk_length, args.p,
                                               args.subw, args.alpha, args.beta)
    vec = dict()
    for fi in files:
        if os.path.isfile(os.path.join(args.input_dir, fi)):
            # 每次只读一张图即可
            # main(os.path.join(args.input_dir, input_old), os.path.join(args.input_dir, input_new),
            #      os.path.join(args.output_dir, output_file), args.num_walks, args.walk_length)
            output_file_rm = os.path.join(output_dir_rm, fi[:-4]+'.txt')
            output_file_emb = os.path.join(output_dir_emb, fi[:-4]+'.emb')
            main(args, rm, vec, fi, output_file_rm, output_file_emb)

    endt = time.process_time()
    print("Run time:", endt-startt)
