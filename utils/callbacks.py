from fastai.callbacks import *
from fastai.vision import *

@dataclass
class MultiTrainSaveModelCallback(TrackerCallback):
    name: str = 'bestmodel'

    def on_train_begin(self, **kwargs):
        super().on_train_begin(**kwargs)
        if not hasattr(self, 'best_global'):
            self.best_global = self.best
            self.cycle = 1
        else:
            self.cycle += 1

    def on_epoch_end(self, epoch, **kwargs):
        current = self.get_monitor_value()
        if current is not None and self.operator(current, self.best):
            self.best = current
            self.learn.save(f'{self.name}_{self.cycle}')
        if current is not None and self.operator(current, self.best_global):
            self.best_global = current
            self.learn.save(f'{self.name}')

@dataclass
class MultiTrainEarlyStoppingCallback(TrackerCallback):
    min_delta: int = 0
    patience: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.operator == np.less:
            self.min_delta *= -1

    def on_train_begin(self, **kwargs):
        if not hasattr(self, 'best'):
            super().on_train_begin(**kwargs)
            self.wait = 0
            self.early_stopped = False

    def on_epoch_end(self, epoch, **kwargs):
        current = self.get_monitor_value()
        if current is None:
            return
        if self.operator(current - self.min_delta, self.best):
            self.best, self.wait = current, 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                print(f'Epoch {epoch}: early stopping')
                self.early_stopped = True
                return True


class PaperspaceLrLogger(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        self.batch = 0
        print('{"chart": "lr", "axis": "batch"}')

    def on_batch_begin(self, train, **kwargs):
        if train:
            self.batch += 1
            print('{"chart": "lr", "x": %d, "y": %.4f}' % (self.batch, self.learn.opt.lr))
