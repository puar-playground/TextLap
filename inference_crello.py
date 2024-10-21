import torch
import datasets
from tqdm import tqdm
import json
import os
import numpy as np
from IPython.display import Image, display
import random
from PIL import Image
from utils.render import convert_transparent_to_white, draw_images_on_canvas, combine_images
from utils.prompt2float import prompt_float
from utils.model import TextLap_multimodal
from datasets import load_dataset

# Set seeds for Python, NumPy, and PyTorch
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If you're using GPU


if __name__=="__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TextLap_multimodal(model='puar-playground/TextLap-Graphic', device=device)

    # Load the dataset
    test_dataset = load_dataset("puar-playground/crello-cap", split='test')
    meta = test_dataset[1]
    size = meta['size']
    id = 'identity_' + meta['id']

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Use Phi-3-Vision to write caption for all elements
    # caption_list = []
    # for i, (img, ocr, label) in enumerate(zip(meta['images'], meta['text_list'], meta['label_list'])):
    #     caption = model.get_element_caption(img, label, ocr)
    #     caption_list.append(caption)

    # # Get Prompt
    # prompt_element, label_list_num = prompt_float(meta['caption_list'][1:], meta['label_list'][1:], meta['bbox_list'][1:], meta['size'])
    # W, H = meta['size']
    # prompt = (f'Please arrange the layout of elements on a canvas with a width of {W} and a height of {H}. '
    #           f'layout elements with descriptions are given in CSS format:\n') + prompt_element + '\nPlease generate the bounding box in JSON format.'
    # print(prompt)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Load pre-computed caption and sampled relationship constraint for elements from the dataset
    val_data = json.load(open('./finetune_data/InstLap_crello_test_float.json', 'r'))
    float_prompt_dict = {x['id']: x['conversations'][0]['value'] for x in val_data}
    prompt = float_prompt_dict[id]
    print(prompt)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Generate Layout using TextLap
    layout_raw = model.get_layout(prompt)
    layout = layout_raw.replace('\\', '')
    layout = json.loads(layout)
    print(layout)

    # Visualize true layout
    true_layout = draw_images_on_canvas(meta['bbox_list'], meta['images'], canvas_size=size, resize=False)
    display(true_layout)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Visualize generated layout
    layout_box_list = list(layout.values())
    layout_box_list = [[int(x1 * size[0]), int(y1 * size[1]), int(x2 * size[0]), int(y2 * size[1])] for [x1, y1, x2, y2] in layout_box_list]
    generated_layout = draw_images_on_canvas(layout_box_list, meta['images'][1:], canvas_size=size, resize=False)
    display(generated_layout)

    combine_images(true_layout, generated_layout, f'./Graphic_layout_result.png')
            