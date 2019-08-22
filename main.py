import argparse
import numpy as np
import networkx as nx
import random_walks
from news2vec import newsfeature2vec
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

def parse_args():
    '''
    Parses the News2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run News2vec.")

    parser.add_argument('--input', nargs='?', default='cora/cora.edge',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='cora/a0.3_0',
                        help='Embeddings path')

    parser.add_argument('--map', nargs='?', default='cora/cora.map',
                        help='Map indice to nodes')

    parser.add_argument('--group', nargs='?', default='cora/group.txt',
                        help='node categories')

    parser.add_argument('--dimensions', type=int, default=100,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 100.')

    parser.add_argument('--num-walks', type=int, default=3,
                        help='Number of walks per source. Default is 3 for Cora.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--a', type=float, default=0.3,
                        help='the unlabeled/test ratio. Default is 0.3.')

    parser.add_argument('--b', type=float, default=10,
                        help='the coefficient of loss2. Default is 10')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--un', dest='unsupervised', action='store_true',
                        help='Boolean specifying (un/semi)supervised. Default is semi-supervised.')
    parser.add_argument('--semi', dest='semi_supervised', action='store_false')
    parser.set_defaults(unsupervised=False)

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()

def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G

def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    if args.unsupervised:
        map_dict = {}
        mapf = pd.read_csv(args.map, index_col=[0], names=['id'])
        for i in mapf.index:
            map_dict[str(i)] = mapf.loc[i, 'id']

        walks1 = list()
        for walk in walks:
            walks1.append(list(map(lambda x: map_dict[str(x)], walk)))
        print(walks1[0][:100])
        newsfeature2vec(walks1, args.output, args.map, args.group, embedding_size=args.dimensions,
                        skip_window=args.window_size, unsupervised=True)
    else:
        mapf = pd.read_csv(args.map, index_col=[0], names=['id'])
        labelf = pd.read_csv(args.group, sep=' ', index_col=[0], names=['label'])
        data = mapf.join(labelf).dropna()
        data['label'] = data['label'].astype(int)
        lb = LabelBinarizer()
        lb.fit(data['label'].values)
        np.random.seed(5)
        shuffled = np.random.permutation(data.index)
        mask_num = int(len(data) * args.a)
        mask_id = shuffled[-mask_num:]
        mask = np.ones([len(data)])
        for m in range(len(data)):
            if data.index[m] in mask_id:
                mask[m] = 0
        data['mask'] = mask
        map_dict = {}
        label_dict = {}
        mask_dict = {}
        for i in data.index:
            map_dict[str(i)] = data.loc[i, 'id']
            label_dict[str(i)] = data.loc[i, 'label']
            mask_dict[str(i)] = data.loc[i, 'mask']

        walks1 = list()
        label_walks = list()
        mask_walks = list()
        for walk in walks:
            walks1.append(list(map(lambda x: map_dict[str(x)], walk)))
            label_walks.append(list(map(lambda x: label_dict[str(x)], walk)))
            mask_walks.append(list(map(lambda x: mask_dict[str(x)], walk)))
        print(walks1[0][:100])
        print(label_walks[0][:100])
        print(mask_walks[0][:100])
        newsfeature2vec(walks1, args.output, args.map, args.group, embedding_size=args.dimensions,
                        skip_window=args.window_size, label_walks=label_walks,
                        mask_walks=mask_walks, unsupervised=False, lb=lb, temp_data=data, coeff=args.b)


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph()
    G = random_walks.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    print(len(walks),walks[0])
    learn_embeddings(walks)


if __name__ == "__main__":

    args = parse_args()
    main(args)
