from trees import InternalTreebankNode, LeafTreebankNode, load_trees, InternalParseNode

import copy
import random
random.seed(42)

'''setting'''
# n = 3
# parent_label = None
# frequent_threshold = 0.005


def count_ngram(trees_data, n=None, parent_label=None):
    num_ngram, ngram_dist = dict(), dict()
    patterns = []
    total_num = 0
    # for i_tree in tqdm.trange(len(trees_data)):
    for i_tree in range(len(trees_data)):
        if isinstance(trees_data[i_tree], InternalTreebankNode):
            tree = trees_data[i_tree]
        elif isinstance(trees_data[i_tree], InternalParseNode):
            tree = trees_data[i_tree].convert()
        else:
            raise ValueError("{} is not a pre-defined node type".format(type(trees_data[i_tree])))
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
                    tmp_curr_position = curr_node[1]
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
                        # string.punctuations, but removed "-" and "%"
                        if len(set('!"#$&\'*+,./:;<=>?@[\\]^_`{|}~').intersection(set(curr_ngram))) > 0:
                            continue
                        if any([_ in curr_ngram for _ in ["-LRB-","-RRB-"]]):
                            continue
                        if not (pos_signal and constituent_signal):
                            continue
                        if curr_ngram not in num_ngram:
                            num_ngram[curr_ngram] = 1
                        else:
                            num_ngram[curr_ngram] += 1
                        try:
                            assert tmp_start_position < tmp_end_position
                        except AssertionError:
                            import pdb
                            pdb.set_trace()
                        curr_patterns.append([tmp_start_position, tmp_end_position, curr_ngram])
                        total_num += 1
        patterns.append(curr_patterns)
    for key, value in num_ngram.items():
        ngram_dist[key] = value/total_num
    # print(total_num)
    return num_ngram, ngram_dist, patterns


def get_frequent_patterns(trees_data, n=3, parent_label=None, frequent_threshold=0.0, pattern_ratio_threshold=None, pattern_num_threshold=None):

    assert pattern_num_threshold != None or pattern_ratio_threshold != None
    assert pattern_ratio_threshold > 0 or pattern_num_threshold > 0

    # get label_dist
    num_labels, label_dist, patterns = {}, {}, {}
    frequent_labels = []
    for i, tree_data in enumerate(trees_data):
        num_labels[i], label_dist[i], patterns[i] = count_ngram(trees_data[i], n=n, parent_label=parent_label)
        for kk, vv in label_dist[i].items():
            if vv >= frequent_threshold and kk not in frequent_labels:
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
        if (pattern_ratio_threshold != None and tmp_ > pattern_ratio_threshold) \
            or (pattern_num_threshold != None and len(frequent_labels) >= pattern_num_threshold):
            break
        frequent_labels.append(_[0])
        tmp_ += _[1]
        y_.append(tmp_)
    return frequent_labels, patterns
