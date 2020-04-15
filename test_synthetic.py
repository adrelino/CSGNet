"""
Contains code to start the visualization process.
"""
import json
import os

import numpy as np
import torch
from torch.autograd.variable import Variable

from src.evaluator import Evaluator
from src.Models.models import ParseModelOutput
from src.utils import read_config
from src.utils.generators.mixed_len_generator import MixedGenerateData
from src.utils.train_utils import chamfer

config = read_config.Config("config_synthetic.yml")
model_name = config.pretrain_modelpath.split("/")[-1][0:-4]

data_labels_paths = {3: "data/synthetic/one_op/expressions.txt",
                     5: "data/synthetic/two_ops/expressions.txt",
                     7: "data/synthetic/three_ops/expressions.txt",
                     9: "data/synthetic/four_ops/expressions.txt",
                     11: "data/synthetic/five_ops/expressions.txt",
                     13: "data/synthetic/six_ops/expressions.txt"}
# first element of list is num of training examples, and second is number of
# testing examples.
proportion = config.proportion  # proportion is in percentage. vary from [1, 100].
dataset_sizes = {
    3: [30000, 50 * proportion],
    5: [110000, 500 * proportion],
    7: [170000, 500 * proportion],
    9: [270000, 500 * proportion],
    11: [370000, 1000 * proportion],
    13: [370000, 1000 * proportion]
}

generator = MixedGenerateData(data_labels_paths={3:data_labels_paths[3], 5:data_labels_paths[5]},
                              batch_size=config.batch_size,
                              canvas_shape=config.canvas_shape)

assert len(generator.unique_draw) == 400
data_labels_paths = {3:data_labels_paths[3]}
dataset_sizes = {3:dataset_sizes[3]}

max_len = max(data_labels_paths.keys())
parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1,
                          max_len, config.canvas_shape)

# total size according to the test batch size.
total_size = 0
config.test_size = sum(dataset_sizes[k][1] for k in dataset_sizes.keys())
for k in dataset_sizes.keys():
    test_batch_size = config.batch_size
    total_size += (dataset_sizes[k][1] // test_batch_size) * test_batch_size

over_all_CD = {}
Pred_Prog = []
Targ_Prog = []
metrics = {}
programs_tar = {}
programs_pred = {}

evaluator = Evaluator(config)

#for jit in [False]:
jit = False
with torch.no_grad():
    total_CD = 0
    test_gen_objs = {}
    programs_tar[jit] = []
    programs_pred[jit] = []

    for k in data_labels_paths.keys():
        test_gen_objs[k] = {}
        test_batch_size = config.batch_size
        test_gen_objs[k] = generator.get_test_data(
            test_batch_size,
            k,
            num_train_images=dataset_sizes[k][0],
            num_test_images=dataset_sizes[k][1],
            jitter_program=jit)

    for k in dataset_sizes.keys():
        test_batch_size = config.batch_size
        for i in range(dataset_sizes[k][1] // test_batch_size):
            print(k, i, dataset_sizes[k][1] // test_batch_size)
            data_, labels = next(test_gen_objs[k])
            pred_images, pred_prog = evaluator.test(data_, parser, max_len)
            target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
            labels = Variable(torch.from_numpy(labels)).cuda()
            targ_prog = parser.labels2exps(labels, k)

            programs_tar[jit] += targ_prog
            programs_pred[jit] += pred_prog
            distance = chamfer(target_images, pred_images)
            total_CD += np.sum(distance)

    over_all_CD[jit] = total_CD / total_size

metrics["chamfer"] = over_all_CD
print(metrics, model_name)
print(over_all_CD)

results_path = "trained_models/results/{}/".format(model_name)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open("trained_models/results/{}/{}".format(model_name, "pred_prog.org"), 'w') as outfile:
    json.dump(programs_pred, outfile)

with open("trained_models/results/{}/{}".format(model_name, "tar_prog.org"), 'w') as outfile:
    json.dump(programs_tar, outfile)

with open("trained_models/results/{}/{}".format(model_name, "top1_metrices.org"), 'w') as outfile:
    json.dump(metrics, outfile)