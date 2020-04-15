import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.autograd.variable import Variable

from .Models.models import Encoder
from .Models.models import ImitateJoint
from .utils.train_utils import prepare_input_op, chamfer, cosine_similarity

def compute_batch_metrics(target_images, pred_images):
    cd = chamfer(target_images, pred_images)
    iou = np.sum(np.logical_and(target_images, pred_images),(1, 2)) / np.sum(np.logical_or(target_images, pred_images),(1, 2))
    cos = cosine_similarity(target_images, pred_images)
    return cd, iou, cos

def show_pair(target_images, pred_images, target_expressions, pred_expressions, cd, iou, cos, condition):
    N = len(target_images)
    assert(N == len(pred_images) and N == len(target_expressions) and N == len(pred_expressions))
    assert(N == len(cd) and N == len(iou) and N == len(cos))

    for i in range(N):
        if condition(cd[i],iou[i],cos[i],target_expressions[i],pred_expressions[i]):
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.imshow(target_images[i], cmap="Greys")
            ax1.set_title('t: {}'.format(target_expressions[i]))
            ax2.imshow(pred_images[i], cmap="Greys")
            m = 'cd: {:.3f} iou: {:.3f} cos: {:.3f}'.format(cd[i],iou[i],cos[i])
            ax2.set_title('p: {} \n{}'.format(pred_expressions[i],m))
            plt.show()
            input("press a key")

#from shapenet_generator test_gen
def image_batch_expand(mini_batch):
    mini_batch = np.expand_dims(mini_batch, 1)
    return np.expand_dims(mini_batch, 0).astype(np.float32)

class Evaluator():
    def __init__(self, config):
        encoder_net = Encoder()
        encoder_net.cuda()
        self.imitate_net = ImitateJoint(hd_sz=config.hidden_size,
                                   input_size=config.input_size,
                                   encoder=encoder_net,
                                   mode=config.mode,
                                   num_draws=400, #len(generator.unique_draw),
                                   canvas_shape=config.canvas_shape)
        self.imitate_net.cuda()
        if config.preload_model:
            print("pre loading model")
            pretrained_dict = torch.load(config.pretrain_modelpath)
            imitate_net_dict = self.imitate_net.state_dict()
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items() if k in imitate_net_dict
            }
            imitate_net_dict.update(pretrained_dict)
            self.imitate_net.load_state_dict(imitate_net_dict)
        self.config = config
        self.imitate_net.eval()

    def test(self, data_, parser, max_len):
        labels = np.zeros((self.config.batch_size, max_len), dtype=np.int32)
        one_hot_labels = prepare_input_op(labels,400)#len(generator.unique_draw))
        one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
        data = Variable(torch.from_numpy(data_)).cuda()
        test_output = self.imitate_net.test([data, one_hot_labels, max_len])
        pred_images, correct_prog, pred_prog = parser.get_final_canvas(
            test_output,
            if_just_expressions=False,
            if_pred_images=True)
        return pred_images, pred_prog

    def test2(self, image_batch, parser, max_len):
        return self.test(image_batch_expand(image_batch), parser, max_len)