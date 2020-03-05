import time
from math import sqrt

import torch
import argparse
import copy
from torch.autograd import Variable
from pathlib import Path

from recallModels.DeepRecommender.layers import input_layers
from recallModels.DeepRecommender.models.DeepAutoEncoder import *

parser = argparse.ArgumentParser(description='RecoEncoder')

parser.add_argument('--lr', type=float, default=0.005, metavar='N',
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, metavar='N',
                    help='L2 weight decay')
parser.add_argument('--drop_prob', type=float, default=0.5, metavar='N',
                    help='dropout drop probability')
parser.add_argument('--noise_prob', type=float, default=0.5, metavar='N',
                    help='noise probability')
parser.add_argument('--aug_step', type=int, default=-1, metavar='N',
                    help='do data augmentation every X step')
parser.add_argument('--num_epochs', type=int, default=100, metavar='N',
                    help='train epochs')
parser.add_argument('--gpu_ids', type=str, default="1,2,3", metavar='N',
                    help='use gpu device ids')
parser.add_argument('--constrained', action='store_true',
                    help='constrained autoencoder')
parser.add_argument('--skip_last_layer_nl', action='store_true',
                    help='if present, decoder\'s last layer will not apply non-linearity function')
parser.add_argument('--hidden_layers', type=str, default="1024,512,512,128", metavar='N',
                    help='hidden layer sizes, comma-separated')
parser.add_argument('--path_to_train_data', type=str, default="/home/yangjieyu/yjy/RecommendRelative/recallModels/ml-1m/ml.train", metavar='N',
                    help='Path to training data')
parser.add_argument('--path_to_eval_data', type=str, default="/home/yangjieyu/yjy/RecommendRelative/recallModels/ml-1m/ml.valid", metavar='N',
                    help='Path to evaluation data')
parser.add_argument('--non_linearity_type', type=str, default="selu", metavar='N',
                    help='type of the non-linearity used in activations')
parser.add_argument('--save_path', type=str, default="autorec.pt", metavar='N',
                    help='where to save model')
parser.add_argument('--predictions_path', type=str, default="out.txt", metavar='N',
                    help='where to save predictions')

args = parser.parse_args()
print(args)

use_gpu = torch.cuda.is_available()  # global flag
if use_gpu:
    print('GPU is available.')
else:
    print('GPU is not available.')


def eval(encoder, eval_data_layer):
    encoder.eval()
    denom = 0.0
    total_epoch_loss = 0.0
    for i, (eval, src) in enumerate(eval_data_layer.iterate_one_epoch_eval()):
        inputs = Variable(src.cuda().to_dense() if use_gpu else src.to_dense())
        targets = Variable(eval.cuda().to_dense() if use_gpu else eval.to_dense())
        outputs = encoder(inputs)
        loss, num_ratings = MMSELoss(outputs, targets)
        total_epoch_loss += loss.data
        denom += num_ratings.data
    return sqrt(total_epoch_loss / denom)

def main():
    params = dict()
    params['batch_size'] = 64
    params['data_dir'] = args.path_to_train_data
    params['major'] = 'users'
    params['itemIdInd'] = 1
    params['userIdInd'] = 0
    print("Loading training data")
    data_layer = input_layers.UserItemRecDataProvider(params=params)
    print("Data loaded")
    print("Total items found: {}".format(len(data_layer.data.keys())))
    print("Vector dim: {}".format(data_layer.vector_dim))

    print("Loading eval data")
    eval_params = copy.deepcopy(params)
    # must set eval batch size to 1 to make sure no examples are missed
    eval_params['batch_size'] = 64
    eval_params['data_dir'] = args.path_to_eval_data
    eval_data_layer = input_layers.UserItemRecDataProvider(params=eval_params,
                                                          user_id_map=data_layer.userIdMap,
                                                          item_id_map=data_layer.itemIdMap)
    eval_data_layer.src_data = data_layer.data

    encoder = AutoEncoder(hiddens=[data_layer.vector_dim] + [int(l) for l in args.hidden_layers.split(',')],
                                 nl_type=args.non_linearity_type,
                                 is_constrained=args.constrained,
                                 dropout=args.drop_prob,
                                 last_layer_activations=not args.skip_last_layer_nl)

    # path_to_model = Path(args.save_path)
    # if path_to_model.is_file():
    #     print("Loading model from: {}".format(path_to_model))
    #     rencoder.load_state_dict(torch.load(args.save_path))

    print('######################################################')
    print('######################################################')
    print('############# AutoEncoder Model: #####################')
    print(encoder)
    print('######################################################')
    print('######################################################')

    # gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
    # print('Using GPUs: {}'.format(gpu_ids))
    # if len(gpu_ids) > 1:
    #     encoder = nn.DataParallel(encoder,
    #                                device_ids=gpu_ids)
    encoder.train()
    if use_gpu: encoder = encoder.cuda()

    if args.noise_prob > 0:
        dp = nn.Dropout(args.noise_prob)

    optimizer = torch.optim.SGD(encoder.parameters(), lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
    for epoch in range(args.num_epochs):
        e_start_time = time.time()
        encoder.train()
        total_epoch_loss = 0.0
        denom = 0.0
        for ind, inputs in enumerate(data_layer.iterate_one_epoch()):
            inputs = Variable(inputs.cuda().to_dense() if use_gpu else inputs.to_dense())
            optimizer.zero_grad()
            outputs = encoder(inputs)
            loss, num_ratings = MMSELoss(outputs,inputs)
            loss /= num_ratings
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss
            denom += 1

            if args.aug_step > 0:
                for t in range(args.aug_step):
                    inputs = Variable(outputs.data)
                    if args.noise_prob > 0:
                        inputs = dp(inputs)
                    optimizer.zero_grad()
                    outputs = encoder(inputs)
                    loss, num_ratings = MMSELoss(outputs, inputs)
                    loss /= num_ratings
                    loss.backward()
                    optimizer.step()
        e_end_time = time.time()
        print('Total epoch {} finished in {} seconds with TRAINING RMSE loss: {}'
              .format(epoch+1, e_end_time - e_start_time, sqrt(total_epoch_loss / denom)))
        if epoch % 5 == 0 or epoch == args.num_epochs-1:
            eval_loss = eval(encoder, eval_data_layer)
            print('Epoch {} EVALUATION LOSS: {}'.format(epoch+1, eval_loss))


if __name__ == '__main__':
    main()
