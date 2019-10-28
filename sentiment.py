#!/usr/bin/env python

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import ordered_memory
from utils.hinton import plot
from utils.locked_dropout import LockedDropout
from utils.utils import build_tree


class SSTClassifier(nn.Module):
    def __init__(self, args, elmo=None, glove=None):
        super(SSTClassifier, self).__init__()

        self.args = args
        self.padding_idx = args.padding_idx

        ninp = args.emsize
        if ninp > 0:
            self.embedding = nn.Embedding(
                args.ntoken, ninp,
                padding_idx=self.padding_idx,
            )
        else:
            self.embedding = None

        self.elmo = elmo
        if elmo is not None:
            ninp += 1024

        self.glove = glove
        if glove is not None:
            ninp += 300

        self.lockdrop = LockedDropout(dropout=args.dropouti)

        self.encoder = ordered_memory.OrderedMemory(ninp, args.nhid, args.nslot,
                                                    dropout=args.dropout, dropoutm=args.dropoutm,
                                                    bidirection=args.bidirection)

        self.mlp = nn.Sequential(
            nn.Dropout(args.dropouto),
            nn.Linear(args.nhid, args.nhid),
            nn.ReLU(),
            nn.Dropout(args.dropouto),
            nn.Linear(args.nhid, args.nout),
        )

        self.drop_input = nn.Dropout(args.dropouti)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, input):
        if self.elmo is not None:
            input_elmo, input_torchtext = input
        else:
            input_torchtext = input
        mask = (input_torchtext != self.padding_idx)

        emb_list = []
        if self.embedding is not None:
            emb_torchtext = self.embedding(input_torchtext)
            emb_list.append(emb_torchtext)
        if self.glove is not None:
            emb_glove = self.glove(input_torchtext).detach()
            emb_list.append(emb_glove)
        if self.elmo is not None:
            emb_elmo = self.elmo(input_elmo)
            assert (mask.long() == emb_elmo['mask']).all()
            emb_elmo = emb_elmo['elmo_representations'][0]
            emb_list.append(emb_elmo)
        emb = torch.cat(emb_list, dim=-1)

        emb.transpose_(0, 1)
        mask.transpose_(0, 1)
        emb = self.lockdrop(emb)

        output = self.encoder(emb, mask)

        output = self.mlp(output)

        return output

    @staticmethod
    def load_model(input_path):
        state = torch.load(input_path)
        print('Loading model from %s' % input_path)
        model = SSTClassifier(state['args'])
        model.load_state_dict(state['state_dict'])
        return model

    def save(self, output_path):
        state = dict(args=self.args,
                     state_dict=self.state_dict())
        torch.save(state, output_path)

    def set_pretrained_embeddings(self, ext_embeddings, ext_word_to_index, word_to_index, finetune=False):
        assert hasattr(self, 'embedding')
        embeddings = self.embedding.weight.data.cpu().numpy()
        for word, index in word_to_index.items():
            if word in ext_word_to_index:
                embeddings[index] = ext_embeddings[ext_word_to_index[word]]
        embeddings = torch.from_numpy(embeddings).to(self.embedding.weight.device)
        self.embedding.weight.data.set_(embeddings)
        self.embedding.weight.requires_grad = finetune


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


###############################################################################
# Training code
###############################################################################


def evaluate(data_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    total_datapoints = 0
    for batch, data in enumerate(data_iter):
        sents = data.text
        lbls = data.label
        count = lbls.shape[0]
        lin_output = model(sents)
        total_loss += torch.sum(
            torch.argmax(lin_output, dim=1) == lbls
        ).float().data
        total_datapoints += count

    return total_loss.item() / total_datapoints


def train():
    # Turn on training mode which enables dropout.
    total_loss = 0
    total_acc = 0
    start_time = time.time()
    for batch, data in enumerate(train_iter):
        sents = data.text
        lbls = data.label

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
                '| {:5d}/{:5d} batches '
                '| lr {:05.5f} | ms/batch {:5.2f} '
                '| loss {:5.2f} | acc {:0.2f}'.format(
                    epoch,
                    batch, len(train_iter),
                    optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval,
                    total_loss.item() / args.log_interval,
                    total_acc.item() / args.log_interval))
            total_loss = 0
            total_acc = 0
            start_time = time.time()
        ###
        batch += 1


def generate_parse():
    from nltk import Tree
    from utils import evalb

    batch = []
    pred_tree_list = []
    targ_tree_list = []

    def process_batch():
        nonlocal batch, pred_tree_list, targ_tree_list
        idx = TEXT.process([example['sents'] for example in batch], device=hidden[0].device)

        model(idx)

        probs = model.encoder.probs
        distance = torch.argmax(probs, dim=-1)
        distance[0] = args.nslot
        probs = probs.data.cpu().numpy()

        for i, example in enumerate(batch):
            sents = example['sents']
            sents_tree = example['sents_tree']
            depth = distance[:, i]

            parse_tree = build_tree(depth, sents)

            if len(sents) <= 100:
                pred_tree_list.append(parse_tree)
                targ_tree_list.append(sents_tree)

            if i == 0:
                for j in range(len(sents)):
                    print('%20s\t%2.2f\t%s' % (sents[j], depth[j], plot(probs[j, i], 1.)))
                print(parse_tree)
                print(sents_tree)
                print('-' * 80)

        batch = []

    np.set_printoptions(precision=2, suppress=True, linewidth=5000, formatter={'float': '{: 0.2f}'.format})

    model.eval()
    hidden = model.encoder.init_hidden(1)

    fin = open('.data/sst/trees/dev.txt', 'r')
    for line in fin:
        line = line.lower()
        sents_tree = Tree.fromstring(line)
        sents = sents_tree.leaves()
        batch.append({'sents_tree': sents_tree, 'sents': sents})

        if len(batch) == 16:
            process_batch()

    if len(batch) > 0:
        process_batch()

    evalb(pred_tree_list, targ_tree_list, evalb_path='./EVALB')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--fine-grained', action='store_true',
                        help='use fine grained label')
    parser.add_argument('--subtrees', action='store_true',
                        help='use fine subtrees')
    parser.add_argument('--glove', action='store_true',
                        help='use pretrained glove embedding')
    parser.add_argument('--elmo', action='store_true',
                        help='use pretrained elmo')
    parser.add_argument('--bidirection', action='store_true',
                        help='use bidirection model')
    parser.add_argument('--emsize', type=int, default=0,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nslot', type=int, default=15,
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
    parser.add_argument('--dropouti', type=float, default=0.3,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropouto', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropoutm', type=float, default=0.2,
                        help='dropout applied to memory (0 = no dropout)')
    parser.add_argument('--attention', type=str, default='softmax',
                        help='attention method')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--test-only', action='store_true',
                        help='Test only')

    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--name', type=str, default=randomhash + '.pt',
                        help='exp name')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--std', action='store_true',
                        help='use standard LSTM')
    parser.add_argument('--philly', action='store_true',
                        help='Use philly cluster')
    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

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
    from torchtext import data
    from torchtext import datasets
    from torchtext.vocab import GloVe

    # set up fields
    TEXT = data.Field(lower=True, include_lengths=False, batch_first=True)
    LABEL = data.Field(sequential=False, unk_token=None)

    # make splits for data
    filter_pred = None
    if not args.fine_grained:
        filter_pred = lambda ex: ex.label != 'neutral'
    train_set, dev_set, test_set = datasets.SST.splits(
        TEXT, LABEL,
        train_subtrees=args.subtrees,
        fine_grained=args.fine_grained,
        filter_pred=filter_pred
    )

    # build the vocabulary
    if args.glove:
        TEXT.build_vocab(train_set, dev_set, test_set, min_freq=1, vectors=GloVe(name='840B', dim=300))
    else:
        TEXT.build_vocab(train_set, min_freq=2)
    LABEL.build_vocab(train_set)

    # make iterator for splits
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train_set, dev_set, test_set),
        batch_size=args.batch_size,
        device='cuda' if args.cuda else 'cpu'
    )

    args.__dict__.update({'ntoken': len(TEXT.vocab),
                          'nout': len(LABEL.vocab),
                          'padding_idx': TEXT.vocab.stoi['<pad>']})

    if args.elmo:
        from allennlp.modules.elmo import Elmo, batch_to_ids

        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        elmo = Elmo(options_file, weight_file, 1, requires_grad=False, dropout=0)

        torchtext_process = TEXT.process


        def elmo_process(batch, device):
            elmo_tensor = batch_to_ids(batch)
            elmo_tensor = elmo_tensor.to(device=device)
            torchtext_tensor = torchtext_process(batch, device)
            return (elmo_tensor, torchtext_tensor)


        TEXT.process = elmo_process
    else:
        elmo = None

    if args.glove:
        glove = torch.nn.Embedding(args.ntoken, 300, _weight=TEXT.vocab.vectors)
    else:
        glove = None

    model = SSTClassifier(args, elmo=elmo, glove=glove)

    if args.resume:
        model_load(args.name)

    if args.cuda:
        model = model.cuda()

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

    if not args.test_only:
        # Loop over epochs.
        lr = args.lr
        stored_loss = 0.

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, patience=2, threshold=0)
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train()
                val_loss = evaluate(dev_iter)
                test_loss = evaluate(test_iter)

                print('-' * 89)
                print(
                    '| end of epoch {:3d} '
                    '| time: {:5.2f}s '
                    '| valid acc: {:.4f} '
                    '| test acc: {:.4f} '
                    '|\n'.format(
                        epoch,
                        (time.time() - epoch_start_time),
                        val_loss,
                        test_loss
                    )
                )

                if val_loss > stored_loss:
                    model_save(args.name)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss
                print('-' * 89)

                scheduler.step(val_loss)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    model_load(args.name)
    test_loss = evaluate(test_iter)
    val_loss = evaluate(dev_iter)

    try:
        generate_parse()
    except:
        print('Unable to parse')

    print('-' * 89)
    print(
        '| valid acc: {:.4f} '
        '| test acc: {:.4f} '
        '|\n'.format(
            val_loss,
            test_loss
        )
    )
