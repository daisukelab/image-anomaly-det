from dlcliche.utils import *
import atwin
from utils import *
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,
    ISONoise
)


def album_tfm():
    return Compose([
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.0, rotate_limit=180, p=.8),
    RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ISONoise(p=0.5),
], p=1)


opt = Options(("--dataroot /mnt/dataset/mvtec_ad/original/zipper "
               "--project mvtecad_zipper "
               "--load_size 420 "
               "--crop_size 384 "
               "--suffix .png "
               "--n_epochs 50 "
               "--lr 0.003 "
               "--num_threads 12 "
               "--backbone resnet18 "
              )).parse()


det = atwin.AnoTwinAD(opt.project, opt.work,
                      suffix=opt.suffix, 
                      resize=opt.load_size,
                      size=opt.crop_size,
                      batch_size=opt.batch_size,
                      workers=opt.num_threads,
                      dataset_cls=atwin.DefectOnBlobDataset,
                      train_album_tfm=album_tfm(),
                      anomaly_color=not opt.anomaly_gray,
)

ORG_DATA = Path(opt.dataroot)

good_files = sorted(ORG_DATA.glob('train/good/*.png'))
det.add_good_samples(good_files)

det.train_setup()

criterion = nn.CrossEntropyLoss()
optimizer = det.optimizer(kind='sgd', lr=0.003, weight_decay=0.9) #opt.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

result = atwin.train_model(det, criterion, optimizer, scheduler, det.dl,
                           num_epochs=opt.n_epochs, device=det.device)

det.save_model('best_model', weights=result['best_weights'])
det.save_model('last_model', weights=result['last_weights'])
