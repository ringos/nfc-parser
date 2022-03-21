import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import json

# import random
# random.seed(42)

try:
    from .trees import InternalTreebankNode, LeafTreebankNode, load_trees
except ModuleNotFoundError:
    from trees import InternalTreebankNode, LeafTreebankNode, load_trees

def load_compatitble_data(path):
    '''
    load pattern_children json file
    '''
    with open(path, 'r', encoding='utf-8') as f:
        pattern_children, individual_labels = json.load(f)
    return pattern_children

def get_multi_ngram_pattern_children(trees, num_ngram="1", parent_label=None):
    """
    here, num_ngram is a str, something like "2,3"
    """
    pattern_children = dict()
    for curr_num_ngram in num_ngram.split(","):
        curr_num_ngram = int(curr_num_ngram)
        tmp_pattern_children = get_pattern_children(trees, num_ngram=curr_num_ngram, parent_label=parent_label)[0]
        pattern_children.update(tmp_pattern_children)
    return pattern_children



def get_pattern_children(trees, num_ngram=1, parent_label=None):
    pattern_children = dict()
    label_vocab = set()
    pattern_vocab = set()
    total_num = 0
    total_label_num = 0
    total_unary_num = 0
    all_ngram_words = []
    for i_tree in range(len(trees)):
        tree = trees[i_tree]
        curr_patterns = []
        pp = InternalTreebankNode("NULL", [tree])
        node_stack = [(pp, 0)]
        while node_stack:
            curr_node = node_stack[0]
            node_stack = node_stack[1:]
            curr_rank = []
            curr_position = curr_node[1]
            for _ in curr_node[0].children:
                tmp_len = len([_ for _ in _.leaves()])
                curr_rank.append((_, curr_position))
                curr_position += tmp_len
            node_stack.extend([_ for _ in curr_rank if isinstance(_[0], InternalTreebankNode)])
            if parent_label == None or curr_node[0].label in parent_label:
                if len(curr_rank) >= num_ngram:
                    for i in range(len(curr_rank)-num_ngram+1):
                        tmp_start_position = curr_rank[i][1]
                        tmp_end_position = curr_rank[i][1]
                        curr_ngram = []
                        curr_ngram_words = []
                        curr_child_labels = dict()
                        constituent_signal, pos_signal = False, False
                        for _ in curr_rank[i:i+num_ngram]:
                            if isinstance(_[0], InternalTreebankNode):
                                curr_ngram.append(_[0].label)
                                constituent_signal = True
                                curr_ngram_words.extend([leaf.word for leaf in _[0].leaves()])
                            else:
                                curr_ngram.append("("+_[0].tag+")")
                                pos_signal = True
                                curr_ngram_words.append(_[0].word)

                            tmp_end_position += len([__ for __ in _[0].leaves()])

                        temp_node_stack = [_[0] for _ in curr_rank[i:i+num_ngram] if isinstance(_[0], InternalTreebankNode)]
                        while temp_node_stack:
                            temp_curr_node = temp_node_stack[0]
                            temp_node_stack = temp_node_stack[1:]
                            temp_node_stack.extend([_ for _ in temp_curr_node.children if isinstance(_, InternalTreebankNode)])
                            if temp_curr_node.label not in curr_child_labels:
                                curr_child_labels[temp_curr_node.label] = 1
                                total_label_num += 1
                            else:
                                curr_child_labels[temp_curr_node.label] += 1
                            

                            # handle unary-chains
                            unary_curr_node = temp_curr_node
                            curr_unary_label = [unary_curr_node.label]
                            while True:
                                if len(unary_curr_node.children) == 1 and isinstance(unary_curr_node.children[0], InternalTreebankNode):
                                    curr_unary_label.append(unary_curr_node.children[0].label)
                                    unary_curr_node = unary_curr_node.children[0]
                                else:
                                    break
                            if len(curr_unary_label) > 1:
                                for m in range(len(curr_unary_label)):
                                    for n in range(m+1, len(curr_unary_label)):
                                        tmp_unary_label = "::".join(curr_unary_label[m:n+1])
                                        if tmp_unary_label not in curr_child_labels:
                                            curr_child_labels[tmp_unary_label] = 1
                                            total_unary_num += 1
                                        else:
                                            curr_child_labels[tmp_unary_label] += 1
                                        
                            label_vocab.add(temp_curr_node.label)



                        curr_ngram = ' '.join(curr_ngram)
                        # if "LeafNode" in curr_ngram:
                        #     continue
                        # string.punctuations, but includes -%
                        if len(set('!"#$&\'*+,./:;<=>?@[\\]^_`{|}~').intersection(set(curr_ngram))) > 0:
                            continue
                        # if any([_ in curr_ngram for _ in ["-LRB-", "-RRB-"]]):
                        #     continue
                        # if not (pos_signal and constituent_signal):
                        #     continue
                        if not constituent_signal:
                            continue
                        if curr_ngram not in pattern_children:
                            pattern_children[curr_ngram] = dict()
                            pattern_children[curr_ngram]["[SELF]"] = 0
                        pattern_children[curr_ngram]["[SELF]"] += 1
                        for key, value in curr_child_labels.items():
                            if key not in pattern_children[curr_ngram]:
                                pattern_children[curr_ngram][key] = 0
                            pattern_children[curr_ngram][key] += value
                        
                        pattern_vocab.add(curr_ngram)
                        
                        curr_patterns.append([tmp_start_position, tmp_end_position, curr_ngram])

                        all_ngram_words.append(curr_ngram_words)

                        total_num += 1
    print("avg num of words in a pattern: {}".format(sum([len(_) for _ in all_ngram_words])/len(all_ngram_words)))
    print("total_num of patterns for {}-gram: {}".format(num_ngram, total_num))
    print("total num of labels (excluding unary labels) for {}-gram: {}".format(num_ngram, total_label_num))
    print("total num of unary for {}-gram: {}".format(num_ngram, total_unary_num))
    return pattern_children, pattern_vocab, label_vocab


def count_ngram(trees, n=None, parent_label=None):
    num_ngram, ngram_dist = dict(), dict()
    patterns = []
    total_num = 0
    # for i_tree in tqdm.trange(len(trees)):
    for i_tree in range(len(trees)):
        tree = trees[i_tree]
        curr_patterns = []
        pp = InternalTreebankNode("NULL", [tree])
        node_stack = [(pp, 0)]
        while node_stack:
            curr_node = node_stack[0]
            node_stack = node_stack[1:]
            curr_rank = []
            curr_position = curr_node[1]
            for _ in curr_node[0].children:
                tmp_len = len([_ for _ in _.leaves()])
                curr_rank.append((_, curr_position))
                curr_position += tmp_len
            node_stack.extend([_ for _ in curr_rank if isinstance(_[0], InternalTreebankNode)])
            if parent_label == None or curr_node[0].label in parent_label:
                if len(curr_rank) >= n:
                    for i in range(len(curr_rank)-n+1):
                        tmp_start_position = curr_rank[i][1]
                        tmp_end_position = curr_rank[i][1]
                        curr_ngram = []
                        constituent_signal, pos_signal = False, False
                        for _ in curr_rank[i:i+n]:
                            if isinstance(_[0], InternalTreebankNode):
                                curr_ngram.append(_[0].label)
                                constituent_signal = True
                            else:
                                curr_ngram.append("("+_[0].tag+")")
                                pos_signal = True
                            tmp_end_position += len([__ for __ in _[0].leaves()])
                        curr_ngram = ' '.join(curr_ngram)
                        # if "LeafNode" in curr_ngram:
                        #     continue
                        # string.punctuations, but includes -%
                        if len(set('!"#$&\'*+,./:;<=>?@[\\]^_`{|}~').intersection(set(curr_ngram))) > 0:
                            continue
                        # if any([_ in curr_ngram for _ in ["-LRB-","-RRB-"]]):
                        #     continue
                        # if not (pos_signal and constituent_signal):
                        #     continue
                        if not constituent_signal:
                            continue
                        if curr_ngram not in num_ngram:
                            num_ngram[curr_ngram] = 1
                        else:
                            num_ngram[curr_ngram] += 1
                        curr_patterns.append([tmp_start_position, tmp_end_position, curr_ngram])
                        total_num += 1
        patterns.append(curr_patterns)
    for key, value in num_ngram.items():
        ngram_dist[key] = value/total_num
    # print(total_num)
    return num_ngram, ngram_dist, patterns

def get_frequent_patterns(trees_data, n="3", parent_label=None, frequent_threshold=0.0, pattern_ratio_threshold=None, pattern_num_threshold=None, total_num_threshold=None):

    """
        frequent_threshold: 
            individual pattern's ratio
            e.g. frequent_threshold = 0.005, then ratio(NP (CC) VP) that is 0.005 can be seen as "frequent"
        pattern_ratio_threshold: 
            accumulative ratio
            e.g. the top 90%
        pattern_num_threshold: 
            individual pattern's num
        total_num_threshold:
            total num of available patterns (top X ones)

    """

    """
    update on 8th, May, 2021:
        num_ngram:
            previous: an integer
            current: a str, something like "2,3"
    """
    
    frequent_threshold = 0.0 if frequent_threshold == -1.0 else frequent_threshold
    pattern_ratio_threshold = None if pattern_ratio_threshold == -1.0 else pattern_ratio_threshold
    pattern_num_threshold = None if pattern_num_threshold == -1 else pattern_num_threshold
    total_num_threshold = None if total_num_threshold == -1 else total_num_threshold

    # assert pattern_num_threshold != None or pattern_ratio_threshold != None

    if pattern_num_threshold == None:
        pattern_num_threshold = 0

    # get label_dist

    all_frequent_labels = []
    all_patterns = dict()

    for curr_num_ngram in n.split(","):
        curr_num_ngram = int(curr_num_ngram)

        num_labels, label_dist, patterns = {}, {}, {}
        frequent_labels = []
        for i, tree_data in enumerate(trees_data):
            num_labels[i], label_dist[i], patterns[i] = count_ngram(trees_data[i], n=curr_num_ngram, parent_label=parent_label)
            for kk, vv in label_dist[i].items():
                if vv >= frequent_threshold and kk not in frequent_labels and num_labels[i][kk] >= pattern_num_threshold:
                    frequent_labels.append(kk)
        for key, value in label_dist.items():
            for kk in frequent_labels:
                if kk not in value:
                    value[kk] = 0.0

        tmp_dist = [[label, label_dist[list(label_dist.keys())[0]][label]] for label in frequent_labels]
        tmp_dist = sorted(tmp_dist, key=lambda x:x[1], reverse=True)
        # frequent_labels = [_[0] for _ in tmp_dist]

        frequent_labels = []
        tmp_ = 0.0
        y_ = []
        for _ in tmp_dist:
            ########################################################################################################
            if (pattern_ratio_threshold != None and tmp_ > pattern_ratio_threshold):
                # or (pattern_num_threshold != None and _[1] < pattern_num_threshold):
                # or (pattern_num_threshold != None and len(frequent_labels) >= pattern_num_threshold):
                break
            if total_num_threshold != None and len(frequent_labels) >= total_num_threshold:
                break
            frequent_labels.append(_[0])
            tmp_ += _[1]
            y_.append(tmp_)
        

        all_frequent_labels.extend(frequent_labels)
        if len(all_patterns.keys()) == 0:
            all_patterns = patterns
        else:
            for key, value in patterns.items():
                for j in range(len(value)):
                    all_patterns[key][j].extend(value[j])

    return all_frequent_labels, all_patterns

if __name__ == "__main__":
    num_ngram = int(sys.argv[1])
    
    input_paths = {
        "ptb-train": "/data/senyang/parsing/self-attentive-parser/data/02-21.10way.clean",
        "ptb-dev": "/data/senyang/parsing/self-attentive-parser/data/22.auto.clean",
        "ptb-test": "/data/senyang/parsing/self-attentive-parser/data/23.auto.clean",
        "ctb-5.1-train": "/data/senyang/parsing/self-attentive-parser/data/ctb_5.1/ctb.train",
        "ctb-5.1-dev": "/data/senyang/parsing/self-attentive-parser/data/ctb_5.1/ctb.dev",
        "ctb-5.1-test": "/data/senyang/parsing/self-attentive-parser/data/ctb_5.1/ctb.test",
        "genia-train": "/data/senyang/parsing/data/ood_corpora/genia/train.gold.stripped",
        "genia-test": "/data/senyang/parsing/data/ood_corpora/genia/test.gold.stripped",
        "genia-dev": "/data/senyang/parsing/data/ood_corpora/genia/dev.gold.stripped",
        "spmrl": "/data/senyang/parsing/self-attentive-parser/data/spmrl/",
    }


    ##############################
    input_key = "spmrl"
    ##############################

    if input_key == "ptb-train-split":
        for i in range(3):
            input_path = input_paths["ptb-train-split"] + str(i+1)
            trees = load_trees(input_path)

            pattern_children, pattern_vocab, label_vocab = get_pattern_children(trees, num_ngram=num_ngram)

            output_path = "/data/senyang/parsing/self-attentive-parser/data/pattern_data/{}_pattern_children_{}-gram.json".format("ptb-train-split"+str(i+1), num_ngram)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([pattern_children, list(label_vocab)], f)

    elif input_key == "spmrl":
        import os

        # languages = ["Basque", "French", "German", "Hebrew", "Hungarian", "Korean", "Polish", "Swedish"]
        languages = ["Basque", "French", "German", "Hungarian", "Korean", "Polish", "Swedish"]

        for lang in languages:
            print("")
            print("Processing {}".format(lang))

            trees = load_trees(os.path.join(input_paths[input_key], lang+".train"))
            pattern_children, pattern_vocab, label_vocab = get_pattern_children(trees, num_ngram=num_ngram)
            output_path = "/data/senyang/parsing/self-attentive-parser/data/pattern_data/{}_pattern_children_{}-gram.json".format(input_key+"-{}".format(lang.lower()), num_ngram)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([pattern_children, list(label_vocab)], f)

    else:
        
        trees = load_trees(input_paths[input_key])
        pattern_children, pattern_vocab, label_vocab = get_pattern_children(trees, num_ngram=num_ngram)
        output_path = "/data/senyang/parsing/self-attentive-parser/data/pattern_data/{}_pattern_children_{}-gram.json".format(input_key, num_ngram)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([pattern_children, list(label_vocab)], f)

    

