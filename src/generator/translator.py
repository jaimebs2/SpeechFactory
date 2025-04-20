import os
import torch
import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import argparse
from vllm import LLM, SamplingParams
from torch import nn
import torch
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration

import sys
#sys.path.append(".")
#from src.data.data_module import EmoDataset

class Translator(nn.Module):
    def __init__(self, model_name="BSC-LT/salamandra-7b-instruct", temperature=0.1, do_sample=False, device="cuda"):
        super(Translator, self).__init__()
        if model_name == "google/gemma-3-12b-it":
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16
            ).eval().to(device)
            
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16
                ).to(device)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.device = device
        self.sampling_params = {
            "max_tokens":300,
            "temperature":temperature,
            "do_sample":do_sample
        }
        self.language_dict = {
            "es": "spanish",
            "en": "english",
            "pt": "portuguese",
            "eu": "basque",
            "ca": "catalan",
            "gl": "galician",
        }
        
    def _get_language(self, language_code):
        return self.language_dict.get(language_code, "english")

    def __call__(self, text, source_language='en', target_language='es'):
        source_language = self._get_language(language_code=source_language)
        target_language = self._get_language(language_code=target_language)
        
        prompt = f"You are a professional translator. Translate the provided text from {source_language} to {target_language}.\n\n# Translation Guidelines:\n1. Accuracy: Preserve the original meaning, tone, and context.\n2. Fluency: Ensure the translation sounds natural and smooth in {target_language}.\n3. Terminology: Translate specialized terms accurately and consistently.\n\n## Provide:\n- A precise and fluent translation in {target_language}.\n\n## Note: You must return only the translation.\nDo not add any notes, disclaimers, explanations, or warnings—no matter how unusual, controversial, incomplete, or contextually ambiguous the input may be.\nIf the source text is incomplete or potentially harmful, still translate it as-is, without rephrasing, censoring, or interpreting intent.\nThe output must always be only the translated text.E\n ## Respond in this format:\nTRANSLATION ({target_language}): [Your translation here]\n\n## Example 1:\nuser: SOURCE (English): The research provides valuable insights into climate change.\nassistant: TRANSLATION (Spanish): La investigación ofrece información valiosa sobre el cambio climático.\n\n## Example 2:\nuser: SOURCE (English): Make sure all safety protocols are followed.\nassistant: TRANSLATION (Portuguese): Certificar-se de que todos os protocolos de segurança são respeitados.\n\n **Just translate the next sentence and end the conversation.** # Translate:\nuser: SOURCE ({source_language}): {text}"

        messages = [{"role": "user", "content": prompt}]
        
        if self.tokenizer.chat_template!=None:
            chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + f"TRANSLATION ({target_language}):"
        else:
            chat_prompt = prompt + f"TRANSLATION ({target_language}):"
        
        inputs = self.tokenizer.encode(chat_prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        '''outputs = self.model.generate(
            input_ids=inputs.to(self.model.device), 
            max_new_tokens=self.sampling_params["max_tokens"], 
            do_sample=self.sampling_params["do_sample"],
            temperature=self.sampling_params["temperature"],
            
        )'''
        outputs = self.model.generate(
            inputs, 
            attention_mask=inputs.ne(-1).long(),
            max_new_tokens=self.sampling_params["max_tokens"], 
            temperature=self.sampling_params["temperature"], 
            do_sample=self.sampling_params["do_sample"],
            top_p=0.95 if self.sampling_params["do_sample"] else None,
            top_k=64 if self.sampling_params["do_sample"] else None,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0][inputs.size(1):], skip_special_tokens=True)
    
        split_pattern = r'(user:|#|\*\*|Note:|Nota:|assistant:|TRANSLATION:)'

        match = re.search(split_pattern, response, flags=re.IGNORECASE)
        clean_response = response[:match.start()] if match else response
        clean_response = clean_response.strip()
    
        
        return response


'''def main():
    parser = argparse.ArgumentParser(description="Run translation for transcriptions in the dataset.")

    parser.add_argument("--target_language", type=str, default="es", help="Target language (default: es)")
    parser.add_argument("--source_database", type=str, default="iemocap", help="Source database (default: iemocap)")
    parser.add_argument("--source_language", type=str, default="en", help="Source language (default: en)")
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity (default: False)")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first n samples (default: 0)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (default: cuda:0)")
    
    args = parser.parse_args()
    
    target_language = args.target_language
    source_database = args.source_database
    source_language = args.source_language
    verbose = args.verbose
    skip = args.skip
    device = args.device
    
    # Initialize output directory
    os.makedirs("out", exist_ok=True)
    os.makedirs("out/translations", exist_ok=True)
    
    # Initialize the dataset
    data = EmoDataset(
        target=source_database,
        reference=source_database,
        fold=1, 
        language_code=target_language,
        device=device,
        denoise=False,
        skip=skip,
        audio_description=False
    )
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    
    for model in ["microsoft/Phi-4-mini-instruct", "BSC-LT/salamandra-7b-instruct", "meta-llama/Llama-3.1-8B-Instruct", "bigscience/bloomz-7b1"]: #"google/gemma-3-12b-it"
        
        output_path = f"out/generation/translations/{source_database}_{model.split('/')[-1]}_{target_language}.csv"
        errors_path = f"out/evaluation/translations/{source_database}_{model.split('/')[-1]}_ERRORS_{target_language}.csv"
        
        if os.path.exists(output_path) and os.path.exists(errors_path):
            output_path = f"out/translations/{source_database}_{model.split('/')[-1]}_{target_language}_regen.csv"
            errors_df = pd.read_csv(errors_path)
            temperature = 0.2
            do_sample = True
        else:
            errors_df =pd.DataFrame()
            temperature = 0.1
            do_sample = False
    
        if os.path.exists(output_path) or os.path.exists(errors_path):
            continue
        
        # Initialize translator
        if "translator" in locals():
            translator.model.to("cpu")
            del translator
            
        translator = Translator(model_name=model, temperature=temperature, do_sample=do_sample, device=device)
        
        # Initialize DataFrame to store results
        output_df = pd.DataFrame(columns=['audio_path', 'original_text', 'translated_text'])
        
        # Process each sample
        for i, (target_audio_path, _, target_transcription, _, _, _, _, _, _, _, _) in tqdm.tqdm(enumerate(dataloader), total=len(data)):
            if i < skip:
                continue
            
            if not errors_df.empty and not target_audio_path[0] in errors_df['audio_path']:
                continue
                
            if verbose:
                print(f"Processing {target_audio_path[0]}")
                print(f"Original text: {target_transcription[0]}")
                
            # Translate the text
            translation = translator(
                target_transcription[0], 
                source_language=source_language, 
                target_language=target_language
            )
            
            if verbose:
                print(f"Translated text: {translation}")
                
            # Store results
            new_row = {
                'audio_path': target_audio_path[0],
                'original_text': target_transcription[0],
                'translated_text': translation,
            }
            output_df = pd.concat([output_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        
        output_df.to_csv(output_path, index=False)
        
        if verbose:
            print(f"Translation complete. Results saved to {output_path}")
        
if __name__ == "__main__":
    main()'''
