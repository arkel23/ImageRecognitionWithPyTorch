import os
import glob
from typing import List
from functools import partial

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import torch.nn as nn
from torchvision import transforms

import cv2
from einops import rearrange
from torchcam import methods


from train import parse_args, build_model



class VisHook:
    def __init__(self,
                 model: nn.Module,
                 model_layers: List[str] = None,
                 vis_mask: str = None,
                 device: str ='cpu',):
        """
        :param model: (nn.Module) Neural Network 1
        :param model_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model = model

        self.device = device

        self.model_info = {}

        self.model_info['Layers'] = []

        self.model_features = {}

        self.model_layers = model_layers

        self.vis_mask = vis_mask

        if vis_mask and 'CAM' in vis_mask:
            self.model = self.model.to(self.device)

            for name, layer in self.model.named_modules():
                if name in self.model_layers:
                    self.model_info['Layers'] += [name]

            self.extractor = methods.__dict__[vis_mask](self.model, target_layer=self.model_info['Layers'][-1], enable_hooks=True)

        else:
            self._insert_hooks()
            self.model = self.model.to(self.device)

            self.model.eval()

        print(self.model_info)


    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model":
            self.model_features[name] = out
        else:
            raise RuntimeError("Unknown model name for _log_layer.")


    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model.named_modules():
            if self.model_layers is not None:
                if name in self.model_layers:
                    self.model_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model", name))
            else:
                self.model_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model", name))


    def inference(self, images):
        """
        Computes the attention rollout for the image(s)
        :param x: (input tensor)
        """
        self.model_features = {}

        images = images.to(self.device)

        if self.vis_mask and 'CAM' in self.vis_mask:
            masks = []
            for i, image in enumerate(images):
                self.extractor._hooks_enabled = True
                self.model.zero_grad()

                preds = self.model(image.unsqueeze(0))

                _, class_idx = torch.max(preds, dim=-1)
                activation_map = self.extractor(class_idx.item(), preds[i:i+1])[0]

                self.extractor.remove_hooks()
                self.extractor._hooks_enabled = False

                masks.append(activation_map)

            return preds, masks

        preds = self.model(images)

        return preds, self.model_features


    def inference_save_vis(self, images, fp):
        preds, features = self.inference(images)
        masks = calc_masks(features, self.vis_mask)

        masked_imgs = []
        for i in range(images.shape[0]):
            img_unnorm = inverse_normalize(images[i].detach().clone())
            img_masked = apply_mask(img_unnorm, masks[i])
            masked_imgs.append(img_masked)

        number_imgs = len(masked_imgs)
        ncols = int(number_imgs ** 0.5)
        nrows = number_imgs // ncols
        number_imgs = ncols * nrows

        fig = plt.figure(figsize=(ncols, nrows))
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols),
                        axes_pad=(0.01, 0.01), direction='row', aspect=True)

        for i, (ax, np_arr) in enumerate(zip(grid, masked_imgs)):
            ax.axis('off')
            ax.imshow(np_arr)

        pil_image = save_images(fig, fp)
            
        return preds, pil_image


def calc_masks(features, vis_mask='GradCAM'):
    if 'CAM' in vis_mask:
        masks = torch.cat(features, dim=0)
        masks = rearrange(masks, 'b ah aw -> b (ah aw)')

    else:
        raise NotImplementedError

    return masks


def inverse_normalize(tensor):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def apply_mask(img, mask):
    '''
    img are pytorch tensors of size C x S1 x S2, range 0-255 (unnormalized)
    mask are pytorch tensors of size (S1 / P * S2 / P) of floating values (range prob -1 ~ 1)
    heatmap combination requires using opencv (and therefore numpy arrays)
    '''
    # convert to numpy array
    mask = rearrange(mask, '(h w) -> h w', h=int(mask.shape[0] ** 0.5))
    mask = mask.cpu().detach().numpy()

    if mask.dtype == 'float16':
        mask = mask.astype('float32')

    mask = cv2.normalize(
        mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    mask = mask.astype('uint8')

    mask = cv2.resize(mask, (img.shape[-1], img.shape[-1]))

    img = rearrange(img.cpu().detach().numpy(), 'c h w -> h w c')
    img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR).astype('uint8')

    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    result = cv2.addWeighted(mask, 0.5, img, 0.5, 0)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    result = Image.fromarray(result)

    return result


def save_images(fig, fp):
    fig.savefig(fp, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print('Saved ', fp)
    return 0


def get_layers(model, model_name):
    if model_name == 'vgg19_bn':
        layers = ['model.features.50']
    elif model_name == 'resnet18':
        layers = ['model.layer4.1.bn2']
    elif 'resnetv2_101' in model_name:
        layers = ['model.stages.3.blocks.2.norm3']
    elif 'beitv2_base_patch16_224_in22k' in model_name or 'deit3_base_patch16_224' in model_name:
        layers = ['model.blocks.11.norm2']
    elif 'convnext' in model_name:
        layers = ['model.stages.3.blocks.2.norm']
    elif 'swin' in model_name:
        layers = ['model.layers.3.blocks.1.norm2']
    else:
        layers = []
        for name, _ in model.named_modules():
            print(name)
            if 'norm' in name or 'bn' in name:
                layers.append(name)
        layers = layers[-1]

    return layers



def search_images(folder):
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')

    # if folder is a file
    if os.path.isfile(folder):
        # if folder is a .txt or .csv with file names
        if os.path.splitext(folder)[1] in ('.txt', '.csv'):
            df = pd.read_csv(folder)
            print('Total image files', len(df))
            return df['dir'].tolist()

        # if folder is a path to an image
        elif any([t.replace('*', '') in os.path.splitext(folder)[1] for t in types]):
            return [folder]

    # else if directory
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(folder, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files', len(files_all))
    return files_all


def build_transform(image_size=224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform


def prepare_inference(args):
    model = build_model(args)

    layers = get_layers(model, args.model_name)

    # prepare model
    # _, _, hook = build_environment_inference(args)

    hook = VisHook(model, layers, args.vis_mask, args.device)

    # prepare inference transform
    transform = build_transform(image_size=args.image_size)

    # Load class names
    dic_classid_classname = None

    if args.dataset_root_path and args.df_classid_classname:
        fp = os.path.join(args.dataset_root_path, args.df_classid_classname)

        if os.path.isfile(fp):
            dic_classid_classname = pd.read_csv(fp, index_col='class_id')['class_name'].to_dict()

    return hook, transform, dic_classid_classname


def prepare_img(fn, args, transform):
    # open img
    img = Image.open(fn).convert('RGB')
    # transform -> convert into batch of 1 -> then move to gpu if needed
    img = transform(img).unsqueeze(0).to(args.device)
    return img


def inference_single(args, hook, img, dic_classid_classname=None,
                     fn=None, save=False):
    # forward image through model
    if fn and save:
        preds, pil_image = hook.inference_save_vis(args, img, save_name=fn)
    else:
        preds, _ = hook.inference(img)

    # print predictions
    preds = preds.squeeze(0)
    for i, idx in enumerate(torch.topk(preds, k=3).indices.tolist()):
        prob = torch.softmax(preds, -1)[idx].item()
        if dic_classid_classname is not None:
            classname = dic_classid_classname[idx]
            out_text = '[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=classname, p=prob*100)
            print(out_text)
        else:
            out_text = '[{idx}] ({p:.2f}%)'.format(idx=idx, p=prob*100)
            print(out_text)
        if i == 0:
            top1_text = out_text

    if save:
        return top1_text, pil_image
    return top1_text


def inference_all(args):
    # search for all images in folder / all images from .txt/csv file / single image path
    files_all = search_images(args.images_path)

    # prepare model and transform for inference
    hook, transform, dic_classid_classname = prepare_inference(args)

    for file in files_all:
        print(file)
        save_fn = os.path.splitext(os.path.split(file)[1])[0]

        # prepare each image for inference: transform and make into batch of 1
        img = prepare_img(file, args, transform)

        # Classify
        inference_single(args, hook, img, dic_classid_classname,
                         save_fn, save=args.vis_mask)


    print('Finished.')
    return 0


def main():
    args = parse_args()
    args.results_dir = 'results_inference'

    inference_all(args)

    return 0


if __name__ == '__main__':
    main()
