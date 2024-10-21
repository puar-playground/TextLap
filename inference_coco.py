import torch
import json
import numpy as np
from transformers import pipeline

def LLM_layout(prompt):
    system_prompt = (f"Given a caption describing the visual layout of an image. "
                 f"Please extract the entities displayed in the image, "
                 f"then plan the layout of all entities according to the positional relation in the caption. "
                 f"Write the layout in the CSS format.")
    input_s = system_prompt + '\nUSER: ' + prompt + '\nASSISTANT: '
    
    generated_text = pipe(input_s, max_length=4096, num_return_sequences=1)[0]['generated_text']
    generated_layout = generated_text.split('\nASSISTANT: ')[-1]
    return generated_layout
    

if __name__=="__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = pipeline("text-generation", model="/home/user/sensei-fs-link/MyFastChat/TextLap/crello_merge_float", device=device)

    test_data = json.load(open('./finetune_data/InstLap_coco_css_layout/InstLap_coco_val_128.json', 'r'))
    
    prompt = test_data[1]['conversations'][0]['value']
    layout = LLM_layout(prompt)
    print(layout)