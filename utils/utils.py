from collections import deque, Counter

import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def build_tree(depth, sen):
    depth = depth
    queue = deque(sen)
    stack = [queue.popleft()]
    head = depth[0] - 1
    for point in depth[1:]:
        d = point - head
        if d > 0:
            for _ in range(d):
                if len(stack) == 1:
                    break
                x1 = stack.pop()
                x2 = stack.pop()
                stack.append([x2, x1])
        if len(queue) > 0:
            stack.append(queue.popleft())
            head = point - 1
    while len(stack) > 2 and isinstance(stack, list):
        x1 = stack.pop()
        x2 = stack.pop()
        stack.append([x2, x1])
    while len(stack) == 1 and isinstance(stack, list):
        stack = stack.pop()
    return stack


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    elif h is None:
        return None
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evalb(pred_tree_list, targ_tree_list, evalb_path="EVALB"):
    import os
    import subprocess
    import re
    import nltk
    import tempfile

    temp_path = tempfile.TemporaryDirectory(prefix="evalb-")
    # temp_path = './test/'
    temp_file_path = os.path.join(temp_path.name, "pred_trees.txt")
    temp_targ_path = os.path.join(temp_path.name, "true_trees.txt")
    temp_eval_path = os.path.join(temp_path.name, "evals.txt")

    print("Temp: {}, {}".format(temp_file_path, temp_targ_path))
    temp_tree_file = open(temp_file_path, "w")
    temp_targ_file = open(temp_targ_path, "w")

    for pred_tree, targ_tree in zip(pred_tree_list, targ_tree_list):
        def process_str_tree(str_tree):
            return re.sub('[ |\n]+', ' ', str_tree)

        def list2tree(node):
            if isinstance(node, nltk.Tree):
                return node
            if isinstance(node, list):
                tree = []
                for child in node:
                    tree.append(list2tree(child))
                return nltk.Tree('<unk>', tree)
            elif isinstance(node, str):
                return nltk.Tree('<word>', [node])

        if re.search(r'[RRB|rrb]- [0-9]', process_str_tree(str(list2tree(targ_tree)))) is not None:
            continue
        temp_tree_file.write(process_str_tree(str(list2tree(pred_tree))) + '\n')
        temp_targ_file.write(process_str_tree(str(list2tree(targ_tree))) + '\n')

    temp_tree_file.close()
    temp_targ_file.close()

    evalb_dir = os.path.join(os.getcwd(), evalb_path)
    evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        temp_targ_path,
        temp_file_path,
        temp_eval_path)

    subprocess.run(command, shell=True)

    with open(temp_eval_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_fscore = float(match.group(1))
                break

    temp_path.cleanup()

    print('-' * 80)
    print('Evalb Prec:', evalb_precision,
          ', Evalb Reca:', evalb_recall,
          ', Evalb F1:', evalb_fscore)

    return evalb_fscore


def remove_bracket(tree):
    if isinstance(tree, str):
        if tree in ['(', ')']:
            return None
        else:
            return tree
    elif isinstance(tree, list):
        new_tree = []
        for child in tree:
            new_child = remove_bracket(child)
            if new_child is not None:
                new_tree.append(new_child)
        if new_tree == []:
            return None
        else:
            while len(new_tree) == 1 and isinstance(new_tree, list):
                new_tree = new_tree[0]
            return new_tree


def char2tree(s):
    stack = []
    for w in s:
        if w == '(':
            stack.append(w)
        elif w == ')':
            node = []
            e = stack.pop()
            while not e == '(':
                node.append(e)
                e = stack.pop()
            node = node[::-1]
            stack.append(node)
        else:
            stack.append(w)
    while len(stack) == 1 and isinstance(stack, list):
        stack = stack[0]
    return stack



def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def ConvertBinaryBracketedSeq(seq):
    T_SHIFT = 0
    T_REDUCE = 1
    T_SKIP = 2

    tokens, transitions = [], []
    for item in seq:
        if item != "(":
            if item != ")":
                tokens.append(item)
            transitions.append(T_REDUCE if item == ")" else T_SHIFT)
    return tokens, transitions
