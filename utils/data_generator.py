import numpy as np
import cv2
import torchvision
from fastai.vision import *
from torch.utils.data.sampler import WeightedRandomSampler
from utils.data_analysis import n_labels, name_label_dict

def shuffle_tfm(image, **kwargs):
    if np.random.rand() < 0.5:
        return image

    dst_image = image.clone()

    shuffled_cells = np.arange(9)
    np.random.shuffle(shuffled_cells)
    cell_size = int(image.shape[1] // 3)

    for i, c in enumerate(shuffled_cells):
        src_x = int(i // 3) * cell_size
        src_y = int(i % 3) * cell_size

        dst_x = int(c // 3) * cell_size
        dst_y = int(c % 3) * cell_size

        cell = image[:, src_x:src_x + cell_size, src_y:src_y + cell_size]

        cell_pil = torchvision.transforms.functional.to_pil_image(cell)
        if np.random.rand() < 0.5:
            cell_pil = torchvision.transforms.functional.hflip(cell_pil)
        if np.random.rand() < 0.5:
            cell_pil = torchvision.transforms.functional.vflip(cell_pil)
        if np.random.rand() < 0.5:
            cell_pil = torchvision.transforms.functional.rotate(cell_pil, angle=np.random.choice([90, 180, 270]))
        cell = torchvision.transforms.functional.to_tensor(cell_pil)

        dst_image[:, dst_x:dst_x + cell_size, dst_y:dst_y + cell_size] = cell

    return dst_image

def cls_wts(label_dict, mu=0.5):
    prob_dict, prob_dict_bal = {}, {}
    max_ent_wt = 1 / 28
    for i in range(28):
        prob_dict[i] = label_dict[i][1] / n_labels
        if prob_dict[i] > max_ent_wt:
            prob_dict_bal[i] = prob_dict[i] - mu * (prob_dict[i] - max_ent_wt)
        else:
            prob_dict_bal[i] = prob_dict[i] + mu * (max_ent_wt - prob_dict[i])
    return prob_dict, prob_dict_bal

def calculate_balance_weights(ds):
    prob_dict, prob_dict_bal = cls_wts(name_label_dict, mu=0.0)
    class_weights = np.array([prob_dict_bal[c] / prob_dict[c] for c in range(28)])

    weights = [np.max(class_weights[np.asarray(list(map(int, y.obj)))]) for y in ds.y]

    return weights, class_weights.tolist()

class HpaImageDataBunch(ImageDataBunch):
    @classmethod
    def create(cls, train_ds: Dataset, valid_ds: Dataset, test_ds: Optional[Dataset] = None, path: PathOrStr = '.',
               bs: int = 96,
               num_workers: int = defaults.cpus, tfms: Optional[Collection[Callable]] = None,
               device: torch.device = None,
               collate_fn: Callable = data_collate, no_check: bool = False) -> 'DataBunch':
        "Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`."
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = bs

        train_weights, _ = calculate_balance_weights(train_ds)
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        dls = [DataLoader(d, b, shuffle=s, sampler=p, drop_last=(s and b > 1), num_workers=num_workers) for d, b, s, p
               in
               zip(datasets, (bs, val_bs, val_bs, val_bs), (False, False, False, False),
                   (train_sampler, None, None, None))]
        return cls(*dls, path=path, device=device, tfms=tfms, collate_fn=collate_fn, no_check=no_check)

def load_image(base_name, image_size):
    r = load_image_channel('{}_red.png'.format(base_name), image_size)
    g = load_image_channel('{}_green.png'.format(base_name), image_size)
    b = load_image_channel('{}_blue.png'.format(base_name), image_size)
    y = load_image_channel('{}_yellow.png'.format(base_name), image_size)
    return np.stack([r, g, b, y], axis=2)

def load_image_channel(file_path, image_size):
    channel = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if channel.shape[0] != image_size:
        channel = cv2.resize(channel, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return channel
