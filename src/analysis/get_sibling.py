import nltk
from nltk.tree import ParentedTree
import json
import sys

def collapse_unary_strip_pos(tree, strip_top=True):
    """Collapse unary chains and strip part of speech tags."""

    def strip_pos(tree):
        if len(tree) == 1 and isinstance(tree[0], str):
            return tree[0]
        else:
            return nltk.tree.Tree(tree.label(), [strip_pos(child) for child in tree])

    collapsed_tree = strip_pos(tree)
    collapsed_tree.collapse_unary(collapsePOS=True, joinChar="::")
    if collapsed_tree.label() in ("TOP", "ROOT", "S1", "VROOT"):
        if strip_top:
            if len(collapsed_tree) == 1:
                collapsed_tree = collapsed_tree[0]
            else:
                collapsed_tree.set_label("")
        elif len(collapsed_tree) == 1:
            collapsed_tree[0].set_label(
                collapsed_tree.label() + "::" + collapsed_tree[0].label())
            collapsed_tree = collapsed_tree[0]
    return collapsed_tree

def _get_labeled_spans(tree, spans_out, start):
    if isinstance(tree, str):
        return start + 1

    # try:
    #     assert len(tree) > 1 or isinstance(
    #     tree[0], str
    # ), "Must call collapse_unary_strip_pos first"
    # except AssertionError:
    #     print(tree.leaves())
    #     raise KeyboardInterrupt

    end = start
    for child in tree:
        end = _get_labeled_spans(child, spans_out, end)
    # Spans are returned as closed intervals on both ends
    spans_out.append((start, end - 1, tree.label()))
    return end

def _get_left_labeled_spans(tree, left_spans_out, left_sib_dict, start):
    if isinstance(tree, str):
        return start + 1

    end = start
    for child in tree:
        end = _get_left_labeled_spans(child, left_spans_out, left_sib_dict, end)
    # Spans are returned as closed intervals on both ends

    if tree.left_sibling() is not None and not isinstance(tree.left_sibling(), str) and start != end-1:
        left_sibling_label = tree.left_sibling().label()
        left_spans_out.append((start, end - 1, left_sibling_label))

        list_left_sibling_label = left_sibling_label.split("::")
        for i in range(len(list_left_sibling_label)):
            curr_sib_label = list_left_sibling_label[i]
            if tree.label() not in left_sib_dict:
                left_sib_dict[tree.label()] =  dict()
                left_sib_dict[tree.label()][curr_sib_label] = 1
            elif curr_sib_label not in left_sib_dict[tree.label()]:
                left_sib_dict[tree.label()][curr_sib_label] = 1
            else:
                left_sib_dict[tree.label()][curr_sib_label] += 1

            for j in range(i+1, len(list_left_sibling_label)):
                curr_sib_label = '::'.join(list_left_sibling_label[i:j])
                if tree.label() not in left_sib_dict:
                    left_sib_dict[tree.label()] =  dict()
                    left_sib_dict[tree.label()][curr_sib_label] = 1
                elif curr_sib_label not in left_sib_dict[tree.label()]:
                    left_sib_dict[tree.label()][curr_sib_label] = 1
                else:
                    left_sib_dict[tree.label()][curr_sib_label] += 1
    return end

def _get_right_labeled_spans(tree, right_spans_out, right_sib_dict, start):
    if isinstance(tree, str):
        return start + 1

    end = start
    for child in tree:
        end = _get_right_labeled_spans(child, right_spans_out, right_sib_dict, end)
    # Spans are returned as closed intervals on both ends

    if tree.right_sibling() is not None and not isinstance(tree.right_sibling(), str) and start != end-1:
        right_sibling_label = tree.right_sibling().label()
        right_spans_out.append((start, end - 1, right_sibling_label))

        list_right_sibling_label = right_sibling_label.split("::")
        for i in range(len(list_right_sibling_label)):
            curr_sib_label = list_right_sibling_label[i]
            if tree.label() not in right_sib_dict:
                right_sib_dict[tree.label()] =  dict()
                right_sib_dict[tree.label()][curr_sib_label] = 1
            elif curr_sib_label not in right_sib_dict[tree.label()]:
                right_sib_dict[tree.label()][curr_sib_label] = 1
            else:
                right_sib_dict[tree.label()][curr_sib_label] += 1

            for j in range(i+1, len(list_right_sibling_label)):
                curr_sib_label = '::'.join(list_right_sibling_label[i:j])
                if tree.label() not in right_sib_dict:
                    right_sib_dict[tree.label()] =  dict()
                    right_sib_dict[tree.label()][curr_sib_label] = 1
                elif curr_sib_label not in right_sib_dict[tree.label()]:
                    right_sib_dict[tree.label()][curr_sib_label] = 1
                else:
                    right_sib_dict[tree.label()][curr_sib_label] += 1
    return end

def get_labeled_spans(tree, left_sib_dict, right_sib_dict):
    """Converts a tree into a list of labeled spans.

    Args:
        tree: an nltk.tree.Tree object

    Returns:
        A list of (span_start, span_end, span_label) tuples. The start and end
        indices indicate the first and last words of the span (a closed
        interval). Unary chains are collapsed, so e.g. a (S (VP ...)) will
        result in a single span labeled "S+VP".
    """
    tree = ParentedTree.convert(collapse_unary_strip_pos(tree))
    

    left_spans_out = []
    _get_left_labeled_spans(tree, left_spans_out, left_sib_dict, start=0)
    # print(left_spans_out)

    right_spans_out = []
    _get_right_labeled_spans(tree, right_spans_out, right_sib_dict, start=0)
    # print(right_spans_out)

    spans_out = []
    _get_labeled_spans(tree, spans_out, start=0)
    # [(0, 0, 'JJ'), (1, 1, 'NNS'), (0, 1, 'NP'), (2, 2, 'VBP'), (3, 3, 'VBN'), (4, 4, 'IN'), (5, 5, 'ADJ'), (6, 6, 'NNS'), (5, 6, 'NP'), (5, 6, 'NP'), (4, 6, 'PP'), (3, 6, 'VP'), (2, 6, 'VP'), (0, 6, 'S'), (7, 7, '.'), (0, 7, 'ROOT')]

    # print(spans_out)
    # print(left_sib_dict)
    # print(right_sib_dict)
    
    return spans_out

if __name__ == "__main__":

    input_paths = {
        "ptb-train": "/data/senyang/parsing/self-attentive-parser/data/02-21.10way.clean",
        "ptb-dev": "/data/senyang/parsing/self-attentive-parser/data/22.auto.clean",
        "ptb-test": "/data/senyang/parsing/self-attentive-parser/data/23.auto.clean",
        "ctb-5.1-train": "/data/senyang/parsing/self-attentive-parser/data/ctb_5.1/ctb.train",
        "genia-train": "/data/senyang/parsing/data/ood_corpora/genia/train.gold.stripped",
        "genia-test": "/data/senyang/parsing/data/ood_corpora/genia/test.gold.stripped",
        "genia-dev": "/data/senyang/parsing/data/ood_corpora/genia/dev.gold.stripped",
        "spmrl": "/data/senyang/parsing/self-attentive-parser/data/spmrl/",
    }

    input_key = sys.argv[1]

    with open(input_paths[input_key], 'r', encoding='utf-8') as f:
        lines = list(f.readlines())

    left_sib_dict, right_sib_dict = {}, {}

    for line in lines:
        ptree = ParentedTree.fromstring(line.strip())
        get_labeled_spans(ptree, left_sib_dict, right_sib_dict)

    print(left_sib_dict)
    print(right_sib_dict)

    output_path = "/data/senyang/parsing/self-attentive-parser/data/sibling_data/{}_sibling_extended.json".format(input_key)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([left_sib_dict, right_sib_dict], f)
