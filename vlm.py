from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor,Qwen2VLModel
from qwen_vl_utils import process_vision_info
import torch

class VLM_EMB(object):

    def __init__(self, model_path="./model/Qwen2-VL-2B-Instruct"):
        # Device configuration
        if torch.backends.mps.is_available():
            self.device_map = "mps"
        else:
            self.device_map = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_path = model_path
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=self.device_map,
        )
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, min_pixels= self.min_pixels, max_pixels= self.max_pixels)
    

    def build_message(self,image_path:str,text:str)->list:
        """
        Build a message for VLM in the video retrieval system.

        :param image_path: Path to the image for which the text prompt is provided.
        :param text: The text prompt describing the context for processing.

        :return: A messages to be processed by the model.
        """
        messages = [
            {"role": "system", "content": "You are a retraval helper for generating better words(vectors) for retriveing videos by given image"},
            {"role": "user",
                "content": [
                    {"type": "image", "image": f"{image_path}"},
                    {"type": "text", "text": f"{text}"},
                ],
            }
        ]
        return messages

    def generate_dense_vector(self, image_path_list:list, text_prompt_list:list,method="mean")->list[list]:
        """
        Generate dense vectors for given image paths and text prompts.

        :param image_path_list: A list of paths to images for which the text prompts are provided.
        :param text_prompt_list: A list of text prompts describing the context for processing.
        
        :return: A list of dense vectors for the given images and text prompts.
        """
        messages = []
        for img_p, txt in zip(image_path_list,text_prompt_list):
            messages.append(self.build_message(img_p,txt))

        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device_map)

        with torch.no_grad():
            output = self.model.forward(**inputs,output_hidden_states=True)
            for hs in output.hidden_states:
                print(hs)

            exit()
            # Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            final_language_hidden = output.hidden_states[-1]

            if method == "mean":

                # Calculate the average embedding 
                average_embeddings = final_language_hidden.mean(dim=1).cpu().detach().to(dtype=torch.float32).numpy().tolist() 
            
            elif method == "last":
                # Use the last embedding 
                average_embeddings = final_language_hidden[-1].cpu().detach().to(dtype=torch.float32).numpy().tolist()

            
            
        
        return average_embeddings
    
    def generate_text(self, image_path_list:list, text_prompt_list:list)->list[str]:
        """
        Generate text for given image paths and text prompts.

        :param image_path_list: A list of paths to images for which the text prompts are provided.
        :param text_prompt_list: A list of text prompts describing the context for processing.

        :return: A list of generated text for the given images and text prompts.
        """

        messages = []
        for img_p, txt in zip(image_path_list,text_prompt_list):
            messages.append(self.build_message(img_p,txt))

        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device_map)
        
        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_texts
    
if __name__ == "__main__":

    my_vlm = VLM_EMB()

    # image_path_list = ["./frames/frame_8990.jpg", "./frames/frame_17980.jpg",]
    # text_prompt_list = ["what is this?","what is this?"]
    image_path_list = ["./frames/frame_600s.jpg",]
    text_prompt_list = ["what is this?"]

    vec_res = my_vlm.generate_dense_vector(image_path_list, text_prompt_list)
    text_res = my_vlm.generate_text(image_path_list, text_prompt_list)
    print(text_res)

    

