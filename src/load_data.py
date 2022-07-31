import pickle
import numpy as np
import random
from tqdm import tqdm
from itertools import islice,combinations
from dppy.finite_dpps import FiniteDPP
import networkx as nx
import collections
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
import torch
from torch.utils import data
import argparse

def build_knowledge_graph(kg, isNew = False):
    def _print_graph_statistic(Graph):
        print('The knowledge graph has been built completely')
        print('The number of nodes is:  ' + str(len(Graph.nodes()))) 
        print('The number of edges is:  ' + str(len(Graph.edges())))
    
    if isNew:
        pair2relation = {}
        for item in kg:
            nodes = kg[item]
            for node in nodes: 
                pair2relation[item,node[0]] = node[1]
                pair2relation[node[0],item] = node[1]

        kg_nodes = []
        kg_edges = collections.defaultdict(list)

        for item in kg:
            kg_nodes.append(item)
            nodes = [each[0] for each in kg[item]]
            for node in nodes:
                kg_edges[item].append(node)

        knowledge_graph = nx.Graph()
        kg_nodes_list = kg_nodes
        kg_edges_list = kg_edges

        for n in kg_nodes_list: knowledge_graph.add_node(n)
        for start in kg_edges_list:
            for end in kg_edges_list[start]:
                knowledge_graph.add_edge(start,end)
                
        f_knowledge_graph = open('./data/knowledge_graph.pkl','wb')
        pickle.dump((knowledge_graph, pair2relation), f_knowledge_graph)
        f_knowledge_graph.close()
    else:
        knowledge_graph = pickle.load(open("./data/knowledge_graph.pkl", "rb"))[0]
        pair2relation = pickle.load(open("./data/knowledge_graph.pkl", "rb"))[1]
    
    _print_graph_statistic(knowledge_graph)
    
    return knowledge_graph, pair2relation

def MinePathsMetaTag(graph, entityid2type, pair2relation, node1, node2, k):
    node2all, node2meta_tag = [],[]
    
    # extract paths from short to longer:
    def _k_shortest_paths(G, source, target, k, weight=None):
        return list(
            islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
        )
    
    if nx.has_path(graph, node1, node2):
        # expand non-shortest paths with k:
        paths = _k_shortest_paths(graph, node1, node2, k)

        # extract paths and meta-tags:
        for i in range(len(paths)):
            t_path, t_meta_tag = [],[]
            for j in range(len(paths[i])):
                if j != len(paths[i]) - 1:
                    t_path.append(paths[i][j]) # add entity
                    t_meta_tag.append(entityid2type[paths[i][j]]) # add entity type
                    if (paths[i][j],paths[i][j + 1]) in pair2relation:
                        t_path.append('r' + str(pair2relation[paths[i][j],paths[i][j + 1]])) # add relation
                        t_meta_tag.append('r' + str(pair2relation[paths[i][j],paths[i][j + 1]]))
                else:
                    t_path.append(paths[i][j])
                    t_meta_tag.append(entityid2type[paths[i][j]])
                    
            t_meta_tag = '->'.join(t_meta_tag)
            node2all.append((t_path,t_meta_tag))
            node2meta_tag.append(t_meta_tag)
    else:
        return [str(node1) + str('->NaP->') + str(node2)], ['B->NaP->B']
            
    return node2all, list(set(node2meta_tag))

def SWING(pairs):
    alpha = 0.5
    
    def _get_uitems_iusers(pairs):
        u_items, i_users = dict(),dict()
        
        # get pairs
        for row in tqdm(pairs):
            u_items.setdefault(row[0], set())
            i_users.setdefault(row[1], set())

            u_items[row[0]].add(row[1])
            i_users[row[1]].add(row[0])
            
        return u_items, i_users
 
    def _cal_similarity(u_items, i_users):
        item_pairs = list(combinations(i_users.keys(), 2))
        item_sim_dict = dict()
        
        # calculating
        cnt = 0
        for (i, j) in tqdm(item_pairs):
            cnt += 1
            user_pairs = list(combinations(i_users[i] & i_users[j], 2))
            result = 0.0
            for (u, v) in user_pairs:
                result += 1 / (alpha + list(u_items[u] & u_items[v]).__len__())

            item_sim_dict[i,j] = result
        
        # Normalization
        _values = [item_sim_dict[key] for key in item_sim_dict]
        _max, _min = max(_values), min(_values)
        _range =  _max - _min
        for key in item_sim_dict: item_sim_dict[key] = (item_sim_dict[key] - _min) / _range
        
        # Statistics  
        print("numbers of node pairs：{}".format(len(u_items)))
        print("numbers of meta-tags：{}".format(len(i_users)))
        print("item pairs length：{}".format(len(item_pairs))) 

        return item_sim_dict
    
    u_items, i_users = _get_uitems_iusers(pairs)
    item_sim_dict = _cal_similarity(u_items, i_users)            
    
    return item_sim_dict,u_items, i_users

def HierarchicalPathsSampling(paths, metaTags, item_sim, k1, k2, hop):
    def _KernelExtract(indexs,sim_dict):
        indexRehased = {}
        Kernel = np.zeros(shape=(len(indexs),len(indexs)))
        count = 0

        for index in indexs:
            indexRehased[count] = index
            count += 1

        for i in range(len(indexs)):
            for j in range(len(indexs)):
                if i == j:
                    Kernel[i][j] = 1.0
                elif (indexs[i],indexs[j]) in sim_dict:
                    Kernel[i][j] = sim_dict[indexs[i],indexs[j]]
                else:
                    Kernel[i][j] = sim_dict[indexs[j],indexs[i]]
        
        if min(np.linalg.eigvals(Kernel)) < 0:
            Kernel = Kernel.T.dot(Kernel)
            
        return Kernel,indexRehased
    
    def _SampleMetaTag(metaTags, k, item_sim):
        def _k_dpp_sampling(k,L):
            from dppy.finite_dpps import FiniteDPP
            DPP = FiniteDPP('likelihood', **{'L': L})
            DPP.flush_samples()
            DPP.sample_exact_k_dpp(size=k)
            return DPP.list_of_samples[0]

        if k < 2:
            return metaTags
        else:
            Kernel,indexRehased = _KernelExtract(metaTags,item_sim)
            return [indexRehased[index] for index in _k_dpp_sampling(k, Kernel)]
    
    def _MaxHopLimit(paths, metaTags, hop, size):
        def sample_path(paths, hop1, hop2):
            tmp = []
            for i in range(len(paths)):
                t_len = int((len(paths[i][0])-1)/2)
                if t_len > hop1 and t_len <= hop2:
                    tmp.append(paths[i])
            return tmp
        
        def sample_tag(metaTags, hop1, hop2):
            tmp = []
            for i in range(len(metaTags)):
                t_len = int(len(metaTags[i].split('->'))/2)
                if t_len > hop1 and t_len <= hop2:
                    tmp.append(metaTags[i])
            return tmp
                    
        if len(paths) < size:
            return paths, metaTags
    
        path,tag = [],[]
        hop1,hop2 = 0, hop

        while(len(path) < size):
            path += sample_path(paths, hop1, hop2)
            tag += sample_tag(metaTags, hop1, hop2)
            hop1 = hop2
            hop2 += 1
            
        return path,tag
    
    def _PathSamping(paths, metaTags, item_sim, k1, k2, hop):
        # max_hop limit:
        paths, metaTags = _MaxHopLimit(paths, metaTags, hop, k2)
        
        selectedMetaTags, candidatePaths, selectedPaths = [],[],[]
        selectedMetaTags = _SampleMetaTag(metaTags, k1, item_sim)
        
        MeteTagCount = {}
        for MetaTag in selectedMetaTags: MeteTagCount[MetaTag] = 0

        for path in paths:
            if path[1] in selectedMetaTags:
                candidatePaths.append(path)
                MeteTagCount[path[1]] += 1
            else:
                continue
        
        if len(candidatePaths) <= k2:
            selected_indexs = np.random.choice(len(paths), size=k2, replace=False)
            return [paths[selected_index][0] for selected_index in selected_indexs]

        for MetaTag in MeteTagCount: MeteTagCount[MetaTag] /= len(candidatePaths)

        while len(selectedPaths) < k2:
            for candidate in candidatePaths:
                if random.random() < MeteTagCount[candidate[1]] and candidate not in selectedPaths: 
                    selectedPaths.append(candidate[0])                    
                if len(selectedPaths) == k2:
                    return selectedPaths
                else:
                    continue
    
    if len(paths) <= k2:
        return paths
    else:
        return _PathSamping(paths, metaTags, item_sim, k1, k2, hop)
    
class Dataset(data.Dataset):
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.name[index]

    def __len__(self):
        return len(self.x)
    
def DataPreprocessing(links, samples, sampleSize, entityid2base_info):
    sample2paths = {}
    for each in samples:
        if len(samples[each]) == sampleSize:
            sample2paths[each] = samples[each]
            
    link2label = {}
    for link in links:
        if str(link[0]) + '|' + str(link[1]) in sample2paths:
            link2label[str(link[0]) + '|' + str(link[1])] = link[2]
    
    feas, label, names = [],[],[]
    for name in sample2paths:
        node1, node2 = name.split('|')[0], name.split('|')[1]
        feas.append(entityid2base_info[int(node1)] + entityid2base_info[int(node2)])
        label.append(link2label[name])
        names.append(name)
    feas = np.asarray(feas)
    print(feas.shape)
    print(np.mean(label))
    
    eval_ratio = 0.2
    test_ratio = 0.2

    n_ratings = len(sample2paths)
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    data_raw1, data_raw2, data_raw3 = {},{},{}
    data_raw1['data'] = (np.array(feas)[train_indices], list(np.array(label)[train_indices]))
    data_raw1['names'] = list(np.array(names)[train_indices])
    
    data_raw2['data'] = (np.array(feas)[eval_indices], list(np.array(label)[eval_indices]))
    data_raw2['names'] = list(np.array(names)[eval_indices])
    
    data_raw3['data'] = (np.array(feas)[test_indices], list(np.array(label)[test_indices]))
    data_raw3['names'] = list(np.array(names)[test_indices])
    
    print(data_raw1['data'][0].shape,data_raw2['data'][0].shape,data_raw3['data'][0].shape)
    print(np.mean(data_raw1['data'][1]),np.mean(data_raw2['data'][1]),np.mean(data_raw3['data'][1]))
    
    return data_raw1, data_raw2, data_raw3

def pipline(args):
    print('load data...')
    f_smc_dict = open("./data/smc_dict-v3.pkl", "rb")
    smc_dict = pickle.load(f_smc_dict)

    knowledge_graph, pair2relation = build_knowledge_graph(smc_dict['kg'], isNew = True)

    # MinePathsMetaTag
    print('Paths and MetaTag Mining...')
    links = smc_dict['sample_dict'][args.dataset].tolist()
    node2all, node2meta_tag = {}, {}
    
    pairs = [link for link in links]
    for pair in tqdm(pairs):
        node1, node2 = pair[0], pair[1]
        key, label = str(node1) + '|' + str(node2), pair[2]
        if key not in node2all:
            node2all[key], node2meta_tag[key] = MinePathsMetaTag(knowledge_graph, smc_dict['entityid2type'], pair2relation, node1, node2, args.K1)
        else:
            continue

    f_node2all = open('./data/node2all.pkl','wb')
    pickle.dump(node2all, f_node2all)
    f_node2all.close()

    f_node2meta_tag = open('./data/node2meta_tag.pkl','wb')
    pickle.dump(node2meta_tag, f_node2meta_tag)
    f_node2meta_tag.close()
    
    
    # MCF
    print('Meta-tag Collaborative Filtering...')
    node2meta_tag_new = {}
    for node in tqdm(node2meta_tag):
        tmp = set()
        for tag in node2meta_tag[node]:
            tmp.add('->'.join(tag.split('->')[::2]))
        node2meta_tag_new[node] = list(tmp)
        
    pairs = []
    for key in tqdm(node2meta_tag_new):
        for value in node2meta_tag_new[key]:
                pairs.append([key,value])
    
    item_sim_dict, u_items, i_users = SWING(pairs)
    
    # DHPS
    print('DPPs-induced Hierarchical Paths Sampling...')
    sampleSize = args.K2
    hop = args.hop
    sample2paths = {}
    
    for link in links:
        key = str(link[0]) + '|' + str(link[1])
        k2 = sampleSize
        
        if len(node2meta_tag_new[key]) < 20:
            k1 = int(len(node2meta_tag_new[key]) / 2)
        else:
            k1 = int(len(node2meta_tag_new[key])**0.5)

        sample2paths[key] = HierarchicalPathsSampling(node2all[key],node2meta_tag_new[key], item_sim_dict, k1, k2, hop)

    # Path2Vec
    documents = []
    for sample in sample2paths:
        for index, items in enumerate(sample2paths[sample]):
            documents.append(TaggedDocument([str(item) for item in items], [str(sample) + ':' + str(index)]))

    path_model = Doc2Vec(documents, dm=1, vector_size=64, window=5, min_count=2, epochs=10, workers=8)

    path2vec = {}
    for sample in sample2paths:
        if len(sample2paths[sample]) == sampleSize:
            for index, items in enumerate(sample2paths[sample]):
                path2vec[str(sample) + ':' + str(index)] = path_model.docvecs[str(sample) + ':' + str(index)]
        else:
            for index, items in enumerate(sample2paths[sample]):
                path2vec[str(sample) + ':' + str(index)] = path_model.docvecs[str(sample) + ':' + str(index)]
            tmp = path_model.docvecs[str(sample) + ':' + str(index)]
            for cnt in range(sampleSize - index -1):
                path2vec[str(sample) + ':' + str(cnt+len(sample2paths[sample]))] = tmp
    
    f_path2vec= open('./data/path2vec.pkl','wb')
    pickle.dump(path2vec, f_path2vec)
    f_path2vec.close()
                
    # Build_Dataset
    print('Dataset Building...')
    np.random.seed(20220630)

    batch_size = args.bs
    entityid2base_info = smc_dict['entityid2base_info']
    train_raw, val_raw, test_raw = DataPreprocessing(links, sample2paths, sampleSize, entityid2base_info)

    train_dataset = Dataset(train_raw['data'][0], train_raw['data'][1], train_raw['names'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = Dataset(val_raw['data'][0], val_raw['data'][1], val_raw['names'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Dataset(test_raw['data'][0], test_raw['data'][1], test_raw['names'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    f_sample_dict= open('./data/dataset.pkl','wb')
    sample_dict = {'train_raw':train_loader, 'val_raw':valid_loader, 'test_raw':test_loader}
    pickle.dump(sample_dict, f_sample_dict)
    f_sample_dict.close()
                
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='专用设备制造业', help='dataset name')
parser.add_argument('--K1', type=int, default=1000, help='number of paths to be sampled')
parser.add_argument('--K2', type=int, default=10, help='size of the sampled path subset')
parser.add_argument('--hop', type=int, default=5, help='max hop')
parser.add_argument('--bs', type=int, default=128, help='batch_size')

def main():
    args = parser.parse_args()
    pipline(args)

if __name__ == '__main__':
    main()
