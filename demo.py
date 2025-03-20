import os
import numpy as np
import gradio as gr

from train import parse_args, yaml_config_hook
from inference import prepare_inference, prepare_img, inference_single


def demo(file_path, vis_mask='GradCAM', dataset='abide'):
    args = parse_args()

    args.vis_mask = vis_mask

    if dataset == 'abide':
        args.cfg = os.path.join('configs', 'abide.yaml')
        args.ckpt_path = os.path.join('results', 'abide_resnet18_0', 'last.pth')

    if args.cfg:
        config = yaml_config_hook(os.path.abspath(args.cfg))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)


    # prepare model and transform for inference
    hook, transform, dic_classid_classname = prepare_inference(args)

    # prepare each image for inference: transform and make into batch of 1
    img = prepare_img(file_path, args, transform)

    save_fp = os.path.join(args.results_dir, 'temp.jpg')
    top1_text, masked_image = inference_single(hook, img, dic_classid_classname,
                                            save_fp, save=True)

    masked_image = np.array(masked_image)

    return top1_text, masked_image


title = 'AI for Medical Image Analysis'
description = 'Image classification demo for "AI for Medical Image Analysis" (AIMIA) Micro-Course on NYCU (2025/03)'
article = '''<p style='text-align: center'>
    AI for Medical Image Analysis 
    </p>'''

inputs = [
    gr.components.Image(type='filepath', label='Input image'),
    gr.components.Radio(value='GradCAM', choices=['CAM', 'GradCAM'],
                        label='Decision interpretation method'),
    gr.components.Radio(value='abide', choices=['abide'],
                        label='Dataset (def: abide)'),
]

outputs = [
    gr.components.Textbox(label='Predicted class and tags'),
    gr.components.Image(label='Decision Heatmap')
]

examples = [
    [os.path.join('samples', 'asd_51160.jpg')],
    [os.path.join('samples', 'td_51110.jpg')],
]

gr.Interface(
    demo, inputs, outputs, title=title, description=description,
    article=article, examples=examples).launch(debug=True, share=True)
