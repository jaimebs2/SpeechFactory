import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration
import torch
from scipy import stats

def load_translations(translation_csv):
    """Load translations from CSV file."""
    if not os.path.exists(translation_csv):
        return pd.DataFrame()
    
    print(f"Loading translations from {translation_csv}")
    df = pd.read_csv(translation_csv)
    return df

class TranslationEvaluator:
    def __init__(self, model_name="BSC-LT/salamandra-7b-instruct", do_sample=False, temperature=0.1, device="cuda"):
        print(f"Loading model {model_name}...")
        if model_name == "google/gemma-3-12b-it":
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                device_map=device
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.do_sample = do_sample
        self.temperature = temperature
        self.language_dict = {
            "es": "Spanish",
            "en": "English",
            "pt": "Portuguese",
            "eu": "Basque",
            "ca": "Catalan",
            "gl": "Galician",
        }
        
    def _get_language_name(self, language_code):
        return self.language_dict.get(language_code, language_code)
        
    def evaluate_translation(self, source_text, translation, source_lang="en", target_lang="es"):
        """Evaluate translation quality using LLM."""
        source_language = self._get_language_name(source_lang)
        target_language = self._get_language_name(target_lang)
        
        prompt = f"You are a professional translation evaluator. Assess the quality of the following {source_language} to {target_language} translation. \n # Evaluate the translation on: \n 1. Accuracy: Does it preserve the original meaning? \n 2. Fluency: Is it natural in {target_language}? \n 3. Terminology: Are specialized terms translated correctly? \n\n ## Provide: \n - A boolean (TRUE, FALSE) to indicate if the translation is acceptable. \n\n ## Respond in this format: \n TRANSLATION ACCEPTABLE: TRUE \n\n ## Example 1: \n user: SOURCE (english): I want to learn more about the universe. \n TRANSLATION (spanish): Quiero aprender más sobre el universo. \n\n assistant: TRANSALTION ACCEPTABLE: TRUE \n\n ## Example 2: \n user: SOURCE (english): Oh, I see what you mean. \n TRANSLATION (spanish): Oh, oh, veo los significados. \n\n assistant: TRANSALTION ACCEPTABLE: FALSE \n\n **Just evaluate the next translation and end the conversation.** # Translate: \n user: SOURCE ({source_language}): {source_text} \n TRANSLATION ({target_language}): {translation}"

        messages = [{"role": "user", "content": prompt}]
        
        if self.tokenizer.chat_template is not None:
            chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "TRANSALTION ACCEPTABLE:"
        else:
            chat_prompt = prompt + "TRANSALTION ACCEPTABLE:"
        inputs = self.tokenizer.encode(chat_prompt, return_tensors="pt").to(self.device)
        #outputs = self.model.generate(inputs, max_new_tokens=4, temperature=self.temperature, do_sample=self.do_sample )
        outputs = self.model.generate(
            inputs, 
            attention_mask=inputs.ne(-1).long(),
            max_new_tokens=4, 
            temperature=self.temperature, 
            do_sample=self.do_sample,
            top_p=0.95 if self.do_sample else None,
            top_k=64 if self.do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0][inputs.size(1):], skip_special_tokens=True)
        
        if 'TRUE' in response:
            return True
        elif 'FALSE' in response:
            return False
        else:
            return False

def main():
    parser = argparse.ArgumentParser(description="Evaluate translation quality using LLM.")
    parser.add_argument("--source_lang", type=str, default="en", help="Source language code")
    parser.add_argument("--target_lang", type=str, default="es", help="Target language code")
    parser.add_argument("--database", type=str, default="iemocap", help="Database name")
    # parser.add_argument("--model", type=str, default="microsoft/Phi-4-mini-instruct", help="LLM to use for evaluation")
    parser.add_argument("--sample", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on")
    
    args = parser.parse_args()
    
    # Paths
    os.makedirs("out/evaluation/translations", exist_ok=True)
    
    translation_csv = f"out/translations/{args.database}_{args.target_lang}.csv"
    translations_df = load_translations(translation_csv)
    
    regen_csv = f"out/translations/{args.database}_{args.target_lang}_regen.csv"
    regen_df = load_translations(regen_csv)
    
    errors_df = load_translations(f"out/evaluation/translations/{args.database}_ERRORS_{args.target_lang}.csv")
    
    if not regen_df.empty and not errors_df.empty:
        translations_df = regen_df[regen_df['audio_path'].isin(errors_df['audio_path'])]
    
    # Sample if requested
    if args.sample is not None and args.sample < len(translations_df):
        translations_df = translations_df.sample(n=args.sample, random_state=42)
    
    for model in ["microsoft/Phi-4-mini-instruct", "BSC-LT/salamandra-7b-instruct", "meta-llama/Llama-3.1-8B-Instruct", "bigscience/bloomz-7b1"]: #"google/gemma-3-12b-it"
        out_dir = f"out/evaluation/translations/{args.database}_{model.split('/')[-1]}_{args.target_lang}.csv"
        if not os.path.exists(out_dir) or not regen_df.empty:
            # Initialize evaluator
            evaluator = TranslationEvaluator(model_name=model, device=args.device)
            
            # Evaluate translations
            results = []
            
            for _, row in tqdm.tqdm(translations_df.iterrows(), total=len(translations_df), desc="Evaluating translations"):
                source_text = row['original_text']
                translation = row['translated_text']
                audio_path = row['audio_path']
                
                result = evaluator.evaluate_translation(
                    source_text, 
                    translation,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang
                )
                
                results.append({
                    "audio_path": audio_path,
                    "source_text": source_text,
                    "translation": translation,
                    "score": result
                })
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Calculate statistics for scores (could be TRUE, FALSE, or None)
            results_df['score']=np.array([str(score).upper() for score in results_df['score']])
            scores = results_df['score']
            
            total = len(scores)
            scores = scores.str.extract(r'^\s*(TRUE|FALSE)', expand=False)
            
            results_df['score'] = scores

            scores = scores.value_counts()      

            acceptable = scores.get('TRUE', 0)
            unacceptable = scores.get('FALSE', 0)
            no_response = total - acceptable - unacceptable
            
            print(f"Total samples: {total}")
            print(f"Acceptable translations: {acceptable} ({acceptable/total:.2%})")
            print(f"Unacceptable translations: {unacceptable} ({unacceptable/total:.2%})")
            print(f"No response: {no_response} ({no_response/total:.2%})")
            
            # Save results
            if errors_df.empty:
                results_df.to_csv(out_dir, index=False)
            else:
                df = pd.read_csv(out_dir)
                df = pd.concat([results_df, df[~df['audio_path'].isin(errors_df['audio_path'])]])
                df.to_csv(out_dir, index=False)
        else:
            continue
    
    dfs = []
    for model in ["microsoft/Phi-4-mini-instruct", "BSC-LT/salamandra-7b-instruct", "meta-llama/Llama-3.1-8B-Instruct"]: #, "google/gemma-3-12b-it"
        df = pd.read_csv(f"out/evaluation/translations/iemocap_{model.split('/')[-1]}_es.csv")
        dfs.append(df)
        
    merged_df = pd.concat(dfs)

    grouped = merged_df.groupby('audio_path')
    unacceptable = grouped['score'].apply(lambda x: x.value_counts().get(False, 0))
    unacceptable = unacceptable[unacceptable > 1]
    final_df = merged_df[dfs[0]['audio_path'].isin(unacceptable.index)]
    
    final_df.to_csv(f"out/evaluation/translations/iemocap_ERRORS_es.csv", index=False)
    
if __name__ == "__main__":
    main()