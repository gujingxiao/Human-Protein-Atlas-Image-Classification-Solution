from fastai.vision import *
from utils.losses import f1_loss, focal_loss, focal_f1_combined_loss
from utils.data_analysis import protein_stats
from utils.data_generator import HpaImageDataBunch, load_image, shuffle_tfm
from utils.metrices import F1Score, optimize_threshold, acc
from utils.metrices import calculate_categories
from utils.callbacks import MultiTrainEarlyStoppingCallback, MultiTrainSaveModelCallback
from utils.build_models import resnet18, resnet34, resnet50, resnet101
from utils.build_models import seresnext50, seresnext101, seresnet50
from utils.build_models import Xception, Densenet121, Densenet169

torch.backends.cudnn.benchmark = True

input_dir = '../HumanProteinData/'
output_dir = './'
submission_dir = './submission/'
train_data_type = 'combine_data.csv'
base_model_dir = './resnet50_512' # or None
model_name = 'resnet50'

image_size = 512
batch_size = 32
num_cycles_1 = 5
num_cycles_2 = 1
cycle_len = 8
num_class = 28
valid_pct = 0.2
seed = 42
lr = 0.001

def create_image(fn):
    return Image(pil2tensor(PIL.Image.fromarray(load_image(fn, image_size=image_size)), np.float32) / 255.)

class HpaImageItemList(ImageItemList):
    _bunch = HpaImageDataBunch

    def open(self, fn):
        return create_image(fn)

def define_model(model_name):
    if model_name == 'resnet50':
        model_type = resnet50
    elif model_name == 'resnet18':
        model_type = resnet18
    elif model_name == 'resnet34':
        model_type = resnet34
    elif model_name == 'resnet101':
        model_type = resnet101
    elif model_name == 'seresnet50':
        model_type = seresnet50
    elif model_name == 'seresnext50':
        model_type = seresnext50
    elif model_name == 'seresnext101':
        model_type = seresnext101
    elif model_name == 'xception':
        model_type = Xception
    elif model_name == 'densenet121':
        model_type = Densenet121
    elif model_name == 'densenet169':
        model_type = Densenet169
    else:
        model_type = resnet50
        print('Invalid Model Type! Use Default Model [Resnet50] to Continue.')

    return model_type

####################################################
#                                                  #
#             Load Data & Augmentations            #
#                                                  #
####################################################
tfms = get_transforms(flip_vert=True,
                      max_rotate=20,
                      max_zoom=1.2,
                      xtra_tfms=[*zoom_crop(scale=(0.8, 1.2), do_rand=True), TfmPixel(shuffle_tfm)()])

test_images = (HpaImageItemList.from_csv(input_dir, 'sample_submission.csv', folder='test', create_func=create_image))

data = (HpaImageItemList
        .from_csv(input_dir, train_data_type, folder='train', create_func=create_image)
        .random_split_by_pct(valid_pct=valid_pct, seed=seed)
        .label_from_df(sep=' ', classes=[str(i) for i in range(num_class)])
        .transform(tfms)
        .add_test(test_images)
        .databunch(bs=batch_size, num_workers=2)
        .normalize(protein_stats))

####################################################
#                                                  #
#           Build Model & Multi GPUs               #
#                                                  #
####################################################
learn = create_cnn(data, define_model(model_name), pretrained=True, cut=-2, ps=0.5,
                   path=Path(output_dir), loss_func=focal_f1_combined_loss, metrics=[acc, F1Score()])

learn.model = learn.model.cuda()
learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1])

if base_model_dir is not None:
    learn.load(os.path.join(base_model_dir, '{}_model_best_f1'.format(model_name)))

####################################################
#                                                  #
#                    Callbacks                     #
#                                                  #
####################################################
early_stopper = MultiTrainEarlyStoppingCallback(learn, monitor='f1_score', mode='max', patience=cycle_len * 2, min_delta=1e-4)
best_f1_model_saver = MultiTrainSaveModelCallback(learn, monitor='f1_score', mode='max', name='{}_model_best_f1'.format(model_name))

learn.callbacks = [early_stopper, best_f1_model_saver]


####################################################
#                                                  #
#                     Training                     #
#                                                  #
####################################################
print('Image sizes: {}'.format(image_size))
if base_model_dir is None:
    learn.freeze()
    learn.fit(1, lr=lr)

learn.unfreeze()
for c in range(num_cycles_1):
    learn.fit_one_cycle(cycle_len, max_lr=lr)
    if early_stopper.early_stopped:
        break

for c in range(num_cycles_2):
    learn.fit_one_cycle(cycle_len, max_lr=(lr / 2))
    if early_stopper.early_stopped:
        break

print('best f1 score: {:.6f}'.format(best_f1_model_saver.best_global))

####################################################
#                                                  #
#     Do Evaluation and Find Best Threshold        #
#                                                  #
####################################################
print("Load model_best_f1")
learn.load(os.path.join(base_model_dir, '{}_model_best_f1'.format(model_name)))

valid_prediction_logits, valid_prediction_categories_one_hot = learn.TTA(ds_type=DatasetType.Valid)
np.save('{}/{}_valid_prediction_logits.npy'.format(submission_dir, model_name), valid_prediction_logits.cpu().data.numpy())
np.save('{}/{}_valid_prediction_categories.npy'.format(submission_dir, model_name), valid_prediction_categories_one_hot.cpu().data.numpy())
best_threshold, best_score = optimize_threshold(valid_prediction_logits, valid_prediction_categories_one_hot, num_class)
print('best threshold / score: {} / {:.6f}'.format(best_threshold, best_score))

####################################################
#                                                  #
#                   Make Submission                #
#                                                  #
####################################################
test_prediction_logits, _ = learn.TTA(ds_type=DatasetType.Test)
np.save('{}/{}_test_prediction_logits.npy'.format(submission_dir, model_name), test_prediction_logits.cpu().data.numpy())

test_prediction_categories = calculate_categories(test_prediction_logits, best_threshold)
submission_df = pd.read_csv('{}/sample_submission.csv'.format(input_dir), index_col='Id', usecols=['Id'])
submission_df['Predicted'] = [' '.join(map(str, c)) for c in test_prediction_categories]
submission_df.to_csv('{}/{}_submission_class_threshold.csv'.format(submission_dir, model_name))

