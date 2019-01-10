from fastai.vision import *
from sklearn.metrics import f1_score as skl_f1_score
from tqdm import tqdm

def f1_score(prediction_logits, targets, threshold=0.5):
    predictions = torch.sigmoid(prediction_logits)
    binary_predictions = predictions > threshold
    return skl_f1_score(targets, binary_predictions, average='macro')

class F1Score(Callback):
    def on_epoch_begin(self, **kwargs):
        self.prediction_logits = []
        self.targets = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        self.prediction_logits.extend(last_output.cpu().data.numpy())
        self.targets.extend(last_target.cpu().data.numpy())

    def on_epoch_end(self, **kwargs):
        self.metric = f1_score(torch.tensor(self.prediction_logits), torch.tensor(self.targets))

def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()

def optimize_threshold(pred, label, classes):
    rng = np.arange(0.1, 0.7, 0.001)
    f1s = np.zeros((rng.shape[0], classes))
    for j, t in enumerate(tqdm(rng)):
        for i in range(classes):
            p = np.array(torch.sigmoid(pred[:, i]) > t, dtype=np.int8)
            scoref1 = skl_f1_score(label[:, i], p, average='binary')
            f1s[j, i] = scoref1

    T = np.empty(classes)
    for i in range(classes):
        T[i] = rng[np.where(f1s[:, i] == np.max(f1s[:, i]))[0][0]]

    return T, np.mean(np.max(f1s, axis=0))

def one_hot_to_categories(one_hot_categories):
    one_hot_categories_np = one_hot_categories.cpu().data.numpy()
    return [np.squeeze(np.argwhere(p == 1)) for p in one_hot_categories_np]


def calculate_categories(prediction_logits, threshold):
    predictions = torch.sigmoid(prediction_logits)
    predictions_np = predictions.cpu().data.numpy()
    return [np.squeeze(np.argwhere(p > threshold), axis=1) for p in predictions_np]