
import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
import cv2
from transformers import AutoImageProcessor, AutoModel, AutoConfig
import SimpleITK as sitk
import torchvision.transforms as transforms

class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        #self.image_processor = AutoImageProcessor.from_pretrained('microsoft/rad-dino')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_transforms = [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5,  0.5, 0.5))
            ]
        self.data_transforms = transforms.Compose(data_transforms)

    def _parse_image(self, img):
        img = Image.fromarray(img)
        pixel_values = self.image_processor(img)['pixel_values'][0]
        # print(pixel_values.shape) 3 x 518 x 518 for rad-dino 太大了
        return pixel_values

    def resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img


    def get_imgs(self, img_path, scale=256):
        x = cv2.imread(str(img_path), 0)
        x = self.resize_img(x, scale)
        img = Image.fromarray(x).convert("RGB")
        img = self.data_transforms(img)
        return img
    
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report
    
    def parse(self, features):
        to_return = {'id': features['id']}

        # for pretraining task
        if "prior_caption" in features: 
            prior_caption = self.clean_report(features.get("prior_caption", ""))
            to_return['prior_caption'] = prior_caption

        if "current_caption" in features:
            current_caption = self.clean_report(features.get("current_caption", ""))
            to_return['curr_caption'] = current_caption
        
        if "progression" in features:
            to_return['progression'] = features.get("progression", "")

        if "reversed_progression" in features:
            to_return['reversed_progression'] = features.get("reversed_progression", "")

        # chest x-ray images
        if features['image_path']:
            image = self.get_imgs(os.path.join(self.args.base_dir, features['image_path'][0]))
        
        if features['context_image']:
            prior_image = self.get_imgs(os.path.join(self.args.base_dir, features['context_image'][0]))
         
        to_return["image"] = image
        to_return["prior_image"] = prior_image

        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class Pretrain_ParseDataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_pretrain_datasets(args):
    pretrain_dataset = Pretrain_ParseDataset(args)
    return pretrain_dataset




