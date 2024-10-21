from transformers import pipeline
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

class TextLap_multimodal(object):
    def __init__(self, model: str = 'puar-playground/TextLap-Graphic', device: str = 'cuda'):

        self.device = device
        self.model = pipeline("text-generation", model=model, device=device)
        self.caption_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True, torch_dtype="auto", device_map=self.device)
        self.caption_processor = AutoProcessor.from_pretrained("microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True)
        
    def ask_phi3v(self, prompt, image):
    
        generation_args = { 
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }
    
        messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
        prompt_in = self.caption_processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.caption_processor(prompt_in, [image], return_tensors="pt").to(self.device)
        
        generate_ids = self.caption_model.generate(**inputs, eos_token_id=self.caption_processor.tokenizer.eos_token_id, **generation_args)
    
        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.caption_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    
        return response

    def get_element_caption(self, element_image, label, ocr):
        img_white = convert_transparent_to_white(element_image)
        if img_white.mode != 'RGBA':
            img_white = img_white.convert('RGBA')
            
        if 'background' in label:
            caption = self.ask_phi3v('The image provided is the background of a graphic design. '
                                'Please describe the background with less than two sentences.',
                                img_white)
            
        elif ocr == '':
            caption = self.ask_phi3v('The image provided is an element in a graphic design. '
                                'Please describe the element with less than two sentences.',
                                img_white)
        else:
            caption = f'A text that says: "{ocr}"'
            
        return caption

    def get_layout(self, prompt):
        system_prompt = ("Given a caption describing the visual layout of elements in a graphic design. "
                         "Please plan the layout of all entities according to the description in the caption. "
                         "Try to avoid overlapping. "
                         "Write the layout in the JSON format.")
        input_s = system_prompt + '\nUSER: ' + prompt + '\nASSISTANT: '
        
        generated_text = self.model(input_s, max_length=4096, num_return_sequences=1)[0]['generated_text']
        generated_layout = generated_text.split('\nASSISTANT: ')[-1]

        return generated_layout