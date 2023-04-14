import random
import mindspore as ms
from mindspore.dataset import vision, transforms, ImageFolderDataset
from mindspore.communication import get_rank, get_group_size


def TwoCropsTransform(img):
    """ Take two random crops of one image as the query and key. """
    return img, img


def create_dataset(dataset_path, batch_size=256, aug_plus=True, distributed=True):
    if distributed:
        data_set = ImageFolderDataset(dataset_path, num_shards=get_group_size(), shard_id=get_rank())
    else: data_set = ImageFolderDataset(dataset_path)

    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            vision.Decode(to_pil=True),
            vision.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply(
                [vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1)], prob=0.8),  # not strengthened
            vision.RandomGrayscale(0.2),
            transforms.RandomApply(
                [vision.GaussianBlur(3, random.uniform(0.1, 2.0))], prob=0.5),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            vision.Decode(to_pil=True),
            vision.RandomResizedCrop(224, scale=(0.2, 1.)),
            vision.RandomGrayscale(0.2),
            vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.4),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
        ]
    
    # define map operations
    data_set = data_set.map(operations=TwoCropsTransform, input_columns="image",
                            output_columns=["image1", "image2"], column_order=["image1", "image2", "label"])
    data_set = data_set.map(operations=augmentation, input_columns="image1")
    data_set = data_set.map(operations=augmentation, input_columns="image2")

    # apply batch operations
    data_set = data_set.batch(batch_size)

    return data_set # batch, channels, width, height


if __name__ == '__main__':
    traindir = "/data/zihan/imagenet/train"
    data_set = create_dataset(traindir)