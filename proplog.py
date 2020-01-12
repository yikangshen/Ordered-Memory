#!/usr/bin/env python

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import ordered_memory
from utils.utils import build_tree, evalb, remove_bracket, char2tree
from utils.hinton import plot

# from orion.client import report_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='data/propositionallogic/',
                        help='location of the data corpus')
    parser.add_argument('--max-op', type=int, default=6,
                        help='maximum number of operator')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nslot', type=int, default=12,
                        help='number of memory slots')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1.,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouto', type=float, default=0.3,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropoutm', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--test-only', action='store_true',
                        help='Test only')

    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--std', action='store_true',
                        help='use standard LSTM')
    parser.add_argument('--philly', action='store_true',
                        help='Use philly cluster')
    args = parser.parse_args()
    args.tied = True

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################


def model_save(fn):
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'wb') as f:
        # torch.save([model, optimizer], f)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }, f)


def model_load(fn):
    global model, optimizer
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'rb') as f:
        checkpoint = torch.load(f)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        val_loss = checkpoint['loss']


class LogicInference(object):
    def __init__(self, datapath='data/propositionallogic/', maxn=12):
        """maxn=0 indicates variable expression length."""
        self.num2char = ['(', ')',
                         'a', 'b', 'c', 'd', 'e', 'f',
                         'or', 'and', 'not']
        self.char2num = {self.num2char[i]: i
                         for i in range(len(self.num2char))}

        self.num2lbl = list('<>=^|#v')
        self.lbl2num = {self.num2lbl[i]: i
                        for i in range(len(self.num2lbl))}

        self.train_set, self.valid_set, self.test_set = [], [], []
        counter = 0
        for i in range(maxn):
            itrainexample = self._readfile(os.path.join(datapath, "train" + str(i)))
            for e in itrainexample:
                counter += 1
                if counter % 10 == 0:
                    self.valid_set.append(e)
                else:
                    self.train_set.append(e)
                # self.train_set = self.train_set + itrainexample

        for i in range(13):
            itestexample = self._readfile(os.path.join(datapath, "test" + str(i)))
            self.test_set.append(itestexample)

    def _readfile(self, filepath):
        f = open(filepath, 'r')
        examples = []
        for line in f.readlines():
            relation, p1, p2 = line.strip().split('\t')
            p1 = p1.split()
            p2 = p2.split()
            examples.append((self.lbl2num[relation],
                             [self.char2num[w] for w in p1],
                             [self.char2num[w] for w in p2]))
        return examples

    def stream(self, dataset, batch_size, shuffle=False, pad=None):
        if pad is None:
            pad = len(self.num2char)
        import random
        import math
        batch_count = int(math.ceil(len(dataset) / float(batch_size)))

        def shuffle_stream():
            if shuffle:
                random.shuffle(dataset)
            for i in range(batch_count):
                yield dataset[i * batch_size: (i + 1) * batch_size]

        def arrayify(stream, pad):
            for batch in stream:
                batch_lbls = np.array([x[0] for x in batch], dtype=np.int64)
                batch_sent = [x[1] for x in batch] + [x[2] for x in batch]
                max_len = max(len(s) for s in batch_sent)
                batch_idxs = np.full((max_len, len(batch_sent)), pad,
                                     dtype=np.int64)
                for i in range(len(batch_sent)):
                    sentence = batch_sent[i]
                    batch_idxs[:len(sentence), i] = sentence
                yield batch_idxs, batch_lbls

        stream = shuffle_stream()
        stream = arrayify(stream, pad)
        return stream


corpus = LogicInference(datapath=args.data, maxn=args.max_op + 1)

###############################################################################
# Build the model
###############################################################################
###
# if args.resume:
#    print('Resuming model ...')
#    model_load(args.resume)
#    optimizer.param_groups[0]['lr'] = args.lr
#    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
#    if args.wdrop:
#        for rnn in model.rnn.cells:
#            rnn.hh.dropout = args.wdrop
###

ntokens = len(corpus.num2char) + 1
nlbls = len(corpus.num2lbl)


class Classifier(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nout, nslot, dropout, dropouti, dropouto, dropoutm):
        super(Classifier, self).__init__()

        self.padding_idx = ntoken - 1
        self.embedding = nn.Embedding(ntoken, ninp,
                                      padding_idx=self.padding_idx)

        self.encoder = ordered_memory.OrderedMemory(ninp, nhid, nslot,
                                                    dropout=dropout, dropoutm=dropoutm)

        self.mlp = nn.Sequential(
            nn.Dropout(dropouto),
            nn.Linear(4 * nhid, nhid),
            nn.ELU(),
            nn.Dropout(dropouto),
            nn.Linear(nhid, nout),
        )

        self.drop = nn.Dropout(dropouti)

        self.cost = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        batch_size = input.size(1)
        mask = (input != self.padding_idx)
        emb = self.drop(self.embedding(input))
        output = self.encoder(emb, mask)
        self.probs = self.encoder.probs

        clause_1 = output[:batch_size // 2]
        clause_2 = output[batch_size // 2:]
        output = self.mlp(torch.cat([clause_1, clause_2,
                                     clause_1 * clause_2,
                                     torch.abs(clause_1 - clause_2)], dim=1))
        return output


if __name__ == "__main__":
    model = Classifier(
        ntoken=ntokens,
        ninp=args.emsize,
        nhid=args.nhid,
        nout=nlbls,
        nslot=args.nslot,
        dropout=args.dropout,
        dropouti=args.dropouti,
        dropouto=args.dropouto,
        dropoutm=args.dropoutm,
    )

    if args.cuda:
        model = model.cuda()
        # model = model.half()

    params = list(model.parameters())
    total_params = sum(x.size()[0] * x.size()[1]
                       if len(x.size()) > 1 else x.size()[0]
                       for x in params if x.size())
    print('Args:', args)
    print('Model total parameters:', total_params)

    optimizer = torch.optim.Adam(params,
                                 lr=args.lr,
                                 betas=(0, 0.999),
                                 eps=1e-9,
                                 weight_decay=args.wdecay)


###############################################################################
# Training code
###############################################################################

@torch.no_grad()
def valid():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    total_datapoints = 0
    for sents, lbls in corpus.stream(corpus.valid_set, args.batch_size * 2):
        count = lbls.shape[0]
        sents = torch.from_numpy(sents)
        lbls = torch.from_numpy(lbls)
        if args.cuda:
            sents = sents.cuda()
            lbls = lbls.cuda()
        lin_output = model(sents)
        total_loss += torch.sum(
            torch.argmax(lin_output, dim=1) == lbls
        ).float().data
        total_datapoints += count
        accs = total_loss.item() / total_datapoints
    return accs

@torch.no_grad()
def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    model.encoder.OM_forward.nslot = args.nslot * 2

    accs = []
    global_loss = 0
    global_datapoints = 0
    for l in range(13):
        total_loss = 0
        total_datapoints = 0
        for sents, lbls in corpus.stream(corpus.test_set[l], args.batch_size * 2):
            count = lbls.shape[0]
            sents = torch.from_numpy(sents)
            lbls = torch.from_numpy(lbls)
            if args.cuda:
                sents = sents.cuda()
                lbls = lbls.cuda()
            lin_output = model(sents)
            total_loss += torch.sum(
                torch.argmax(lin_output, dim=1) == lbls
            ).float().data.item()
            total_datapoints += count
        accs.append(total_loss / total_datapoints if total_datapoints > 0 else -1)
        global_loss += total_loss
        global_datapoints += total_datapoints

    accs.append(global_loss / global_datapoints)

    model.encoder.OM_forward.nslot = args.nslot
    return accs


def train():
    # Turn on training mode which enables dropout.
    total_loss = 0
    total_acc = 0
    start_time = time.time()
    batch = 0
    for sents, lbls in corpus.stream(corpus.train_set, args.batch_size,
                                     shuffle=True):
        sents = torch.from_numpy(sents)
        lbls = torch.from_numpy(lbls)
        if args.cuda:
            sents = sents.cuda()
            lbls = lbls.cuda()

        model.train()
        optimizer.zero_grad()

        lin_output = model(sents)
        loss = model.cost(lin_output, lbls)
        acc = torch.mean(
            (torch.argmax(lin_output, dim=1) == lbls).float())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += loss.detach().data
        total_acc += acc.detach().data
        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print(
                '| epoch {:3d} '
                '| lr {:05.5f} | ms/batch {:5.2f} '
                '| loss {:5.2f} | acc {:0.2f}'.format(
                    epoch,
                    optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval,
                    total_loss.item() / args.log_interval,
                    total_acc.item() / args.log_interval))
            total_loss = 0
            total_acc = 0
            start_time = time.time()
        ###
        batch += 1

@torch.no_grad()
def genparse():
    model.eval()
    model.encoder.OM_forward.nslot = args.nslot * 2

    np.set_printoptions(precision=2, suppress=True, linewidth=5000, formatter={'float': '{: 0.2f}'.format})
    pred_tree_list = []
    targ_tree_list = []
    for l in range(13):
        for sents, lbls in corpus.stream(corpus.test_set[l], args.batch_size * 2):
            sents = torch.from_numpy(sents)
            if args.cuda:
                sents = sents.cuda()

            # hidden = model.encoder.init_hidden(sents.size(1))
            # emb = model.drop(model.embedding(sents))
            # raw_output, probs_batch, _ = model.encoder(emb, hidden)

            model(sents)
            probs_batch = model.probs

            for i in range(sents.size(1)):
                probs = probs_batch[:, i].view(-1, args.nslot * 2)
                # self.distance = (torch.cumsum(self.probs, dim=-1) < 0.5).sum(dim=-1)

                distance = torch.argmax(probs, dim=-1)
                distance[0] = args.nslot * 2
                sen = [corpus.num2char[x]
                       for x in sents[:, i] if x < len(corpus.num2char)]
                if len(sen) < 2:
                    continue
                depth = distance[:len(sen)]
                probs = probs.data.cpu().numpy()

                parse_tree = remove_bracket(build_tree(depth, sen))
                sen_tree = char2tree(sen)

                pred_tree_list.append(parse_tree)
                targ_tree_list.append(sen_tree)

                if np.random.randint(0, 100) > 0:
                    continue
                print()
                for i in range(len(sen)):
                    print('%5s\t%2.2f\t%s' % (sen[i], distance[i], plot(probs[i], 1)))

                print(' '.join(sen))
                # print(sen_tree)
                print(parse_tree)
                print('')

    evalb(pred_tree_list, targ_tree_list)

    model.encoder.OM_forward.nslot = args.nslot


if __name__ == "__main__":
    # Loop over epochs.
    if not args.test_only:
        lr = args.lr
        stored_loss = 0.

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, patience=2, threshold=0)
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train()
                val_loss = valid()
                test_loss = evaluate()

                print('-' * 89)
                print(
                    '| end of epoch {:3d} '
                    '| time: {:5.2f}s '
                    '| valid acc: {:.2f} '
                    '|\n'.format(
                        epoch,
                        (time.time() - epoch_start_time),
                        val_loss
                    ),
                    ', '.join(str('{:0.2f}'.format(v)) for v in test_loss)
                )

                if val_loss > stored_loss:
                    model_save(args.save)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss
                print('-' * 89)

                scheduler.step(val_loss)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        # Load the best saved model.
    model_load(args.save)

    genparse()

    test_loss = evaluate()
    val_loss = valid()
    print('-' * 89)
    print(
        '| valid acc: {:.2f} '
        '|\n'.format(
            val_loss
        ),
        ', '.join(str('{:0.2f}'.format(v)) for v in test_loss)
    )

    # report_results([dict(
    #     name='val_loss',
    #     type='objective',
    #     value=val_loss)])

