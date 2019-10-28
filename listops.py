import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import ordered_memory
from utils.hinton import plot
from utils.listops_data import load_data_and_embeddings, LABEL_MAP, PADDING_TOKEN, get_batch
from utils.utils import build_tree, char2tree, evalb


class ListOpsModel(nn.Module):
    def __init__(self, args):
        super(ListOpsModel, self).__init__()

        self.args = args
        self.padding_idx = args.padding_idx
        self.embedding = nn.Embedding(args.ntoken, args.ninp,
                                      padding_idx=self.padding_idx)

        self.encoder = ordered_memory.OrderedMemory(args.ninp, args.nhid, args.nslot,
                                                    dropout=args.dropout, dropoutm=args.dropoutm,
                                                    bidirection=args.bidirection)

        self.mlp = nn.Sequential(
            nn.Dropout(args.dropouto),
            nn.Linear(args.nhid * 2 if args.bidirection else args.nhid, args.nout),
        )

        self.drop_input = nn.Dropout(args.dropouti)
        self.drop_output = nn.Dropout(args.dropouto)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, input):
        mask = (input != self.padding_idx).bool()

        emb = self.embedding(input)
        emb.transpose_(0, 1)

        mask.transpose_(0, 1)
        emb = self.drop_input(emb)
        output = self.encoder(emb, mask, output_last=True)
        output = self.mlp(output)
        return output

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
            'loss': test_loss
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
        test_loss = checkpoint['loss']


###############################################################################
# Training code
###############################################################################

@torch.no_grad()
def evaluate(data_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0
    total_datapoints = 0
    for batch, data in enumerate(data_iter):
        batch_data = get_batch(data)
        X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch_data

        X_batch = torch.from_numpy(X_batch).long().to('cuda' if args.cuda else 'cpu')
        y_batch = torch.from_numpy(y_batch).long().to('cuda' if args.cuda else 'cpu')

        lin_output = model(X_batch)
        count = y_batch.shape[0]
        total_loss += torch.sum(
            torch.argmax(lin_output, dim=1) == y_batch
        ).float().data
        total_datapoints += count

    return total_loss.item() / total_datapoints


def train():
    # Turn on training mode which enables dropout.
    model.train()

    total_loss = 0
    total_acc = 0
    start_time = time.time()
    for batch, data in enumerate(training_data_iter):
        # print(data)
        # batch_data = get_batch(next(training_data_iter))
        data, n_batches = data
        batch_data = get_batch(data)
        X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch_data

        X_batch = torch.from_numpy(X_batch).long().to('cuda' if args.cuda else 'cpu')
        y_batch = torch.from_numpy(y_batch).long().to('cuda' if args.cuda else 'cpu')

        optimizer.zero_grad()

        lin_output = model(X_batch)
        loss = model.cost(lin_output, y_batch)
        acc = torch.mean(
            (torch.argmax(lin_output, dim=1) == y_batch).float())
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
                '| {:5d}/ {:5d} batches '
                '| lr {:05.5f} | ms/batch {:5.2f} '
                '| loss {:5.2f} | acc {:0.2f}'.format(
                    epoch,
                    batch,
                    n_batches,
                    optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval,
                    total_loss.item() / args.log_interval,
                    total_acc.item() / args.log_interval))
            total_loss = 0
            total_acc = 0
            start_time = time.time()
        ###
        batch += 1
        if batch >= n_batches:
            break


@torch.no_grad()
def generate_parse(data_iter):
    model.eval()
    np.set_printoptions(precision=2, suppress=True, linewidth=5000, formatter={'float': '{: 0.2f}'.format})
    pred_tree_list = []
    targ_tree_list = []
    crop_count = 0
    total_count = 0
    for batch, data in enumerate(data_iter):
        sents = data['tokens']
        X = np.array([vocabulary[t] for t in data['tokens']])
        # if len(sents) > 100:      # In case Evalb fail to process very long sequences
        #     continue

        X_batch = torch.from_numpy(X).long().to('cuda' if args.cuda else 'cpu')

        model(X_batch[None, :])
        probs = model.encoder.probs
        distance = torch.argmax(probs, dim=-1)
        distance[0] = args.nslot

        total_count += 1
        depth = distance[:, 0]
        probs_k = probs[:, 0, :].data.cpu().numpy()

        try:
            parse_tree = build_tree(depth, sents)
            sen_tree = char2tree(data['sentence'].split())
        except:
            crop_count += 1
            print('Unbalanced datapoint!')
            continue

        pred_tree_list.append(parse_tree)
        targ_tree_list.append(sen_tree)

        if batch % 100 > 0:
            continue
        print(batch)
        for i in range(len(sents)):
            if sents[i] == '<pad>':
                break
            print('%20s\t%2.2f\t%s' % (sents[i], depth[i], plot(probs_k[i], 1)))
        print(parse_tree)
        print(sen_tree)
        print()

    print('Cropped: %d, Total: %d' % (crop_count, total_count))
    evalb(pred_tree_list, targ_tree_list, evalb_path="../EVALB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data', type=str, default='./data/listops',
                        help='location of the data corpus')
    parser.add_argument('--bidirection', action='store_true',
                        help='use bidirection model')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='max sequence length')
    parser.add_argument('--seq_len_test', type=int, default=1000,
                        help='max sequence length')
    parser.add_argument('--no-smart-batching', action='store_true',  # reverse
                        help='batch based on length')
    parser.add_argument('--no-use_peano', action='store_true',
                        help='batch based on length')
    parser.add_argument('--emsize', type=int, default=128,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=128,
                        help='number of hidden units per layer')
    parser.add_argument('--nslot', type=int, default=21,
                        help='number of memory slots')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1.,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--batch_size_test', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropoutm', type=float, default=0.3,
                        help='dropout applied to memory (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.1,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropouto', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
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
    args = parser.parse_args()

    args.smart_batching = not args.no_smart_batching
    args.use_peano = not args.no_use_peano

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
    train_data_path = os.path.join(args.data, 'train_d20s.tsv')
    test_data_path = os.path.join(args.data, 'test_d20s.tsv')
    vocabulary, initial_embeddings, training_data_iter, eval_iterator, training_data_length, raw_eval_data \
        = load_data_and_embeddings(args, train_data_path, test_data_path)
    dictionary = {}
    for k, v in vocabulary.items():
        dictionary[v] = k
    # make iterator for splits
    vocab_size = len(vocabulary)
    num_classes = len(set(LABEL_MAP.values()))
    args.__dict__.update({'ntoken': vocab_size,
                          'ninp': args.emsize,
                          'nout': num_classes,
                          'padding_idx': vocabulary[PADDING_TOKEN]})

    model = ListOpsModel(args)

    if args.cuda:
        model = model.cuda()

    params = list(model.parameters())
    total_params = sum(x.size()[0] * x.size()[1]
                       if len(x.size()) > 1 else x.size()[0]
                       for x in params if x.size())
    total_params_sanity = sum(np.prod(x.size()) for x in model.parameters())
    assert total_params == total_params_sanity
    print("TOTAL PARAMS: %d" % sum(np.prod(x.size()) for x in model.parameters()))
    print('Args:', args)
    print('Model total parameters:', total_params)

    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
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
                test_loss = evaluate(eval_iterator)

                print('-' * 89)
                print(
                    '| end of epoch {:3d} '
                    '| time: {:5.2f}s '
                    '| test acc: {:.4f} '
                    '|\n'.format(
                        epoch,
                        (time.time() - epoch_start_time),
                        test_loss
                    )
                )

                if test_loss > stored_loss:
                    model_save(args.name)
                    print('Saving model (new best validation)')
                    stored_loss = test_loss
                print('-' * 89)

                scheduler.step(test_loss)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    model_load(args.name)
    generate_parse(raw_eval_data)
    test_loss = evaluate(eval_iterator)
    data = {'args': args.__dict__,
            'parameters': total_params,
            'test_acc': test_loss}
    print('-' * 89)
    print(
        '| test acc: {:.4f} '
        '|\n'.format(
            test_loss
        )
    )
