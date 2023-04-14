import mindspore as ms
from mindspore import nn, ops, Parameter
from mindspore.communication import get_rank


class MoCo(nn.Cell):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, teacher_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, args=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # class_num is the output fc dimension
        self.encoder_q = base_encoder(class_num=dim, pretrained=False)
        self.encoder_k = base_encoder(class_num=dim, pretrained=False)

        self.teacher_encoder_q = teacher_encoder(class_num=dim)
        self.teacher_encoder_k = teacher_encoder(class_num=dim)
        
        if mlp:
            # teacher mlp
            dim_mlp = self.teacher_encoder_q.end_point.in_channels
            self.teacher_encoder_q.end_point = nn.SequentialCell([nn.Dense(dim_mlp, dim_mlp), nn.ReLU(), self.teacher_encoder_q.end_point])
            self.teacher_encoder_k.end_point = nn.SequentialCell([nn.Dense(dim_mlp, dim_mlp), nn.ReLU(), self.teacher_encoder_k.end_point])

            # student mlp
            dim_mlp = self.encoder_q.end_point.in_channels
            self.encoder_q.end_point = nn.SequentialCell([nn.Dense(dim_mlp, 2048), nn.ReLU(), nn.Dense(2048, 128)])
            self.encoder_k.end_point = nn.SequentialCell([nn.Dense(dim_mlp, 2048), nn.ReLU(), nn.Dense(2048, 128)])

        # teacher gard
        for param_q, param_k in zip(self.teacher_encoder_q.get_parameters(), 
                                    self.teacher_encoder_k.get_parameters()):
            param_k = param_q.clone()  # initialize
            param_k.requires_grad = False  # not update by gradient
            param_q.requires_grad = False

        # student grad
        for param_q, param_k in zip(self.encoder_q.get_parameters(), 
                                    self.encoder_k.get_parameters()):
            param_k = param_q.clone()  # initialize
            param_k.requires_grad = False  # not update by gradient
            param_q.requires_grad = True

        # create the queue
        self.queue = Parameter(ops.StandardNormal()((dim, K)), name="queue", requires_grad=False)
        self.queue = ops.L2Normalize(axis=0)(self.queue)

        self.queue_ptr = Parameter(ops.Zeros()(1, ms.int64), name="queue_ptr", requires_grad=False)

    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.get_parameters(), 
                                    self.encoder_k.get_parameters()):
            param_k.set_data(param_k.data * self.m + param_q.data * (1 - self.m))
        return

    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = ops.AllGather()(keys)

        batch_size = keys.shape[0]
        print(batch_size)

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        return

    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = ops.AllGather()(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = ops.Randperm()(batch_size_all)

        # broadcast to all gpus
        ops.Broadcast(0)(idx_shuffle)

        # index for restoring
        _, idx_unshuffle = ops.Sort()(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = ops.AllGather()(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def construct(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        #print ("query images shape:", im_q.shape)
        #print ("key images shape:", im_k.shape)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = ops.L2Normalize(axis=1)(q)

        qk = self.encoder_q(im_k)
        qk = ops.L2Normalize(axis=1)(qk)

        # compute key features
        self._momentum_update_key_encoder()  # update the key encoder

        teacher_qk = self.teacher_encoder_q(im_k)
        teacher_qk = ops.L2Normalize(axis=1)(teacher_qk)

        # shuffle for making use of BN
        im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

        k = self.encoder_k(im_k)  # keys: NxC
        k = ops.L2Normalize(axis=1)(k)
        k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        #teacher_k = self.teacher_encoder_k(im_k)
        #teacher_k = nn.functional.normalize(teacher_k, dim=1)
        #teacher_k = self._batch_unshuffle_ddp(teacher_k, idx_unshuffle)

        teacher_q = self.teacher_encoder_q(im_q)  # queries: NxC
        teacher_q = ops.L2Normalize(axis=1)(teacher_q)


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = ops.Einsum('nc,nc->n')((q, k)).expand_dims(-1)
        # negative logits: NxK
        l_neg = ops.Einsum('nc,ck->nk')((q, self.queue.copy()))

        # logits: Nx(1+K)
        logits = ops.Concat(axis=1)(l_pos, l_neg)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = ops.Zeros()((logits.shape[0], ), dtype=ms.int64)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, q, teacher_q, qk, teacher_qk