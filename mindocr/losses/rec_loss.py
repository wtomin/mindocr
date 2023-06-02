import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore.nn.loss.loss import LossBase
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore import ops
import numpy as np

__all__ = ['CTCLoss', 'VisionLANLoss']

# TODO: support label_weights for imbalance data
class CTCLoss(LossBase):
    """
     CTCLoss definition

     Args:
        pred_seq_len(int): the length of the predicted character sequence. For text images, this value equals to W - the width of feature map encoded by the visual bacbkone. This can be obtained by probing the output shape in the network.
            E.g., for a training image in shape (3, 32, 100), the feature map encoded by resnet34 bacbkone is in shape (512, 1, 4), W = 4, sequence len is 4.
        max_label_len(int): the maximum number of characters in a text label, i.e. max_text_len in yaml.
        batch_size(int): batch size of input logits. bs
     """

    def __init__(self, pred_seq_len=26, max_label_len=25, batch_size=32, reduction='mean'):
        super(CTCLoss, self).__init__()
        assert pred_seq_len > max_label_len, 'pred_seq_len is required to be larger than max_label_len for CTCLoss. Please adjust the strides in the backbone, or reduce max_text_length in yaml'
        self.sequence_length = Tensor(np.array([pred_seq_len] * batch_size), mstype.int32)
        label_indices = []
        for i in range(batch_size):
            for j in range(max_label_len):
                label_indices.append([i, j])
        self.label_indices = Tensor(np.array(label_indices), mstype.int64)
        #self.reshape = P.Reshape()
        self.ctc_loss = ops.CTCLoss(ctc_merge_repeated=True)

        self.reduction = reduction
        #print('D: ', self.label_indices.shape)

    # TODO: diff from paddle, paddle takes `label_length` as input too.
    def construct(self, pred: Tensor, label: Tensor):
        '''
        Args:
            pred (Tensor): network prediction which is a
                logit Tensor in shape (W, BS, NC), where W - seq len, BS - batch size. NC - num of classes (types of character + blank + 1)
            label (Tensor): GT sequence of character indices in shape (BS, SL), SL - sequence length, which is padded to max_text_length
        Returns:
            loss value (Tensor)
        '''
        logit = pred
        #T, bs, nc = logit.shape
        #logit = ops.reshape(logit, (T*bs, nc))
        label_values = ops.reshape(label, (-1,))

        loss, _ = self.ctc_loss(logit, self.label_indices, label_values, self.sequence_length)

        if self.reduction=='mean':
            loss = loss.mean()

        return loss

class VisionLANLoss(LossBase):
    def __init__(self, mode='LF_1', weight_res = 0.5, weight_mas = 0.5, reduction='mean', **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        assert mode in ['LF_1', 'LF_2', 'LA']
        self.mode = mode
        self.weight_res = weight_res
        self.weight_mas = weight_mas

    def flatten_label(self, target):
        label_flatten = []
        label_length = []
        for i in range(0, target.shape[0]):
            cur_label = target[i].tolist()
            label_flatten += cur_label[:cur_label.index(0) + 1]
            label_length.append(cur_label.index(0) + 1)
        label_flatten = Tensor(label_flatten, ms.int64)
        label_length = Tensor(label_length, ms.int32)
        return (label_flatten, label_length)

    def _flatten(self, sources, lengths):
        return ops.concat([t[:l] for t, l in zip(sources, lengths)])

    def constrcut(self, predicts, batch):
        text_pre = predicts[0]
        target = batch[1].astype('int64')
        label_flatten, length = self.flatten_label(target)
        text_pre = self._flatten(text_pre, length)
        if self.mode == 'LF_1':
            loss = self.criterion(text_pre, label_flatten)
        else:
            text_rem = predicts[1]
            text_mas = predicts[2]
            target_res = batch[2].astype('int64')
            target_sub = batch[3].astype('int64')
            label_flatten_res, length_res = self.flatten_label(target_res)
            label_flatten_sub, length_sub = self.flatten_label(target_sub)
            text_rem = self._flatten(text_rem, length_res)
            text_mas = self._flatten(text_mas, length_sub)
            loss_ori = self.criterion(text_pre, label_flatten)
            loss_res = self.criterion(text_rem, label_flatten_res)
            loss_mas = self.criterion(text_mas, label_flatten_sub)
            loss = loss_ori + loss_res * self.weight_res + loss_mas * self.weight_mas
        return loss
    
class AttentionLoss(LossBase):
    def __init__(self, reduction='mean'):
        super().__init__()
        # ignore <GO> symbol
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)

    def construct(self, logits, labels):
        labels = labels[:, 1:]  # wihout <GO> symbol
        num_classes = logits.shape[-1]
        logits = ops.reshape(logits, (-1, num_classes))
        labels = ops.reshape(labels, (-1,))
        return self.criterion(logits, labels)


if __name__ == '__main__':
    max_text_length = 23
    nc = 26
    bs = 32
    pred_seq_len  = 24

    loss_fn = CTCLoss(pred_seq_len, max_text_length, bs)

    x = ms.Tensor(np.random.rand(pred_seq_len, bs, nc), dtype=ms.float32)
    label = ms.Tensor(np.random.randint(0, nc,  size=(bs, max_text_length)), dtype=ms.int32)

    loss = loss_fn(x, label)
    print(loss)
