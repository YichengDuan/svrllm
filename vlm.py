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
    
    def generate_dense_vector(self, image_path_list: list, text_prompt_list: list, method="simple_mean",n_layer=4) -> list[list]:
        """
        Generate dense vectors for given image paths and text prompts.

        :param image_path_list: A list of paths to images for which the text prompts are provided.
        :param text_prompt_list: A list of text prompts describing the context for processing.
        :param method: Aggregation method - "simple_mean", "max_pool", "mean_pool", "concat_layers"
        :param n_layer: Number of layers to use for pooling.

        :return: A list of dense vectors for the given images and text prompts.
        """
        
        messages = []
        for img_p, txt in zip(image_path_list, text_prompt_list):
            messages.append(self.build_message(img_p, txt))

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
            output = self.model.forward(**inputs, output_hidden_states=True)

            # Hidden states of the model at the output of each layer plus the optional initial embedding outputs.
            hidden_states = output.hidden_states  # Tuple: (num_layers, batch_size, seq_length, hidden_size)

            # Extract the last hidden layer's hidden states for language
            final_language_hidden = hidden_states[-1]  # Shape: (batch_size, seq_length, hidden_size)

            if method == "simple_mean":
                # Simple mean pooling across all tokens
                average_embeddings = final_language_hidden.mean(dim=1).cpu().detach().to(dtype=torch.float32).numpy().tolist()
                embeddings = average_embeddings

            elif method == "max_pool":
                # Max pooling across all tokens
                max_embeddings = final_language_hidden.max(dim=1)[0].cpu().detach().to(dtype=torch.float32).numpy().tolist()
                embeddings = max_embeddings

            elif method == "mean_pool":
                # Mean pooling with attention mask to exclude padding tokens
                attention_mask = inputs['attention_mask']  # Shape: (batch_size, seq_length)
                mask = attention_mask.unsqueeze(-1).expand(final_language_hidden.size()).float()  # Shape: (batch_size, seq_length, hidden_size)
                sum_hidden = final_language_hidden * mask  # Apply mask
                sum_hidden = torch.sum(sum_hidden, dim=1)  # Sum over tokens
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)  # Avoid division by zero
                mean_pooled = sum_hidden / sum_mask  # Mean pooling
                mean_embeddings = mean_pooled.cpu().detach().to(dtype=torch.float32).numpy().tolist()
                embeddings = mean_embeddings

            elif method == "concat_layers":
                # Combine multiple hidden layers for richer embeddings
                selected_hidden_states = hidden_states[-n_layer:]  # Select last 'num_layers_to_use' layers

                # Stack selected layers: shape (num_layers, batch_size, seq_length, hidden_size)
                stacked_hidden = torch.stack(selected_hidden_states, dim=0)  # Shape: (num_layers, batch_size, seq_length, hidden_size)

                # Apply attention mask to each layer
                attention_mask = inputs['attention_mask']  # Shape: (batch_size, seq_length)
                mask = attention_mask.unsqueeze(0).unsqueeze(-1).expand(stacked_hidden.size()).float()  # Shape: (num_layers, batch_size, seq_length, hidden_size)

                # Apply mask and compute mean pooling for each layer
                masked_hidden = stacked_hidden * mask  # Shape: (num_layers, batch_size, seq_length, hidden_size)
                sum_hidden = torch.sum(masked_hidden, dim=2)  # Sum over tokens: (num_layers, batch_size, hidden_size)
                sum_mask = torch.clamp(mask.sum(dim=2), min=1e-9)  # Sum of masks: (num_layers, batch_size, hidden_size)
                mean_pooled_layers = sum_hidden / sum_mask  # Mean pooling: (num_layers, batch_size, hidden_size)

                # Average the selected layers
                averaged_layers = torch.mean(mean_pooled_layers, dim=0)  # Shape: (batch_size, hidden_size)

                # Normalize the averaged embedding
                normalized_averaged = torch.nn.functional.normalize(averaged_layers, p=2, dim=1)  # Shape: (batch_size, hidden_size)

                # Convert to list
                averaged_embeddings = normalized_averaged.cpu().detach().to(dtype=torch.float32).numpy().tolist()
                embeddings = averaged_embeddings


            else:
                raise ValueError(f"Unknown aggregation method: {method}")

        return embeddings
    
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
    def muti_image_message_builder(self,image_inputs:list,text:str):
        # Messages containing multiple images and a text query
        messages = [
            {"role": "system", "content": "You are a summary helper for describe a story using given images and text transcribe"},
            {"role": "user",
                "content": [],
            }
        ]
        for img_path in image_inputs:
            messages[1]["content"].append({"type": "image", "image": f"{img_path}"})

        # Add the text query to the messages
        messages[1]["content"].append({"type": "text", 
                                       "text": f"Please describe the story behind the image. Do not describe the image itself. The transcription of the provided image is: {text}"})

        return messages
    def muti_image_text(self,image_inputs:list,text:str):
        
        messages = self.muti_image_message_builder(image_inputs,text)

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device_map)

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text

if __name__ == "__main__":

    my_vlm = VLM_EMB()
    
    # image_path_list = ["./frames/frame_8990.jpg", "./frames/frame_17980.jpg",]
    # text_prompt_list = ["what is this?","what is this?"]
    image_path_list = ["./frames/frame_600s.jpg",]
    text_prompt_list = ["what is this?fjdskla;fdjl;sjakl"]
    # ["simple_mean", "max_pool", "mean_pool", "concat_layers"]
    vec_res = my_vlm.generate_dense_vector(image_path_list, text_prompt_list)
    

    

