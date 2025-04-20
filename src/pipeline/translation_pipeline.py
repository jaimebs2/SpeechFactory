# pip install transformers>4.50.0
import os
import argparse
import pandas as pd
import tqdm
import random
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(".")
from src.generator.translator import Translator
from src.data.data_module import EmoDataset
from src.evaluator.translator import TranslationEvaluator

class TranslationPipeline:
    def __init__(
        self,
        target_database,
        target_language,
        source_database,
        source_language,
        device="cuda:0",
        verbose=False,
        skip=0,
        fold=1,
        split='test',
        balance=False,
        sample=False,
        force_balance=False,
        evaluation_attempts_until_threshold_decrease=3,
        increase_max_count=1,
    ):
        self.target_database = target_database
        self.target_language = target_language
        self.source_database = source_database
        self.source_language = source_language
        self.device = device
        self.verbose = verbose
        self.skip = skip
        self.evaluation_attempts_until_threshold_decrease = evaluation_attempts_until_threshold_decrease
        
        # Define LLM models to use
        self.models = [
            "microsoft/Phi-4-mini-instruct",
            "BSC-LT/salamandra-7b-instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "bigscience/bloomz-7b1",
            "google/gemma-3-12b-it",
        ]
        
        # Set up directories
        os.makedirs("out/generation/translations", exist_ok=True)
        os.makedirs("out/evaluation/translations", exist_ok=True)
        
        # Initialize dataset
        self.data = EmoDataset(
            target=self.source_database,
            reference=self.target_database,
            fold=fold,
            language_code=self.target_language,
            device=self.device,
            denoise=False,
            skip=self.skip,
            audio_description=False,
            split=split,
            balance=balance,
            sample=sample,
            force_balance=force_balance,
            increase_max_count=increase_max_count,
        )
        
        self.dataloader = DataLoader(self.data, batch_size=1, shuffle=False)
        
        # Add new path for final translations
        self.final_translations_path = f"out/generation/translations/{self.source_database}_FINAL_{self.target_language}.csv"
        
        self.models_threshold = 0

    def generate_translations(self):
        """Generate translations for all models"""
        if self.verbose:
            print("Starting translation generation...")
            
        errors_path = f"out/evaluation/translations/{self.source_database}_ERRORS_{self.target_language}.csv"
        if os.path.exists(errors_path):
            errors_df = pd.read_csv(errors_path)
        else:
            errors_df = pd.DataFrame()
        
        prev_generated_data = pd.read_csv(f'out/evaluation/translations/{self.source_database}_{self.target_language}.csv')
            
        for model in self.models:
            model_name = model.split('/')[-1]
            if errors_df.empty:
                output_path = f"out/generation/translations/{self.source_database}_{model_name}_{self.target_language}.csv"
            else:
                output_path = f"out/generation/translations/{self.source_database}_{model_name}_{self.target_language}_regen.csv"
            
            # Skip if translations already exist
            '''if os.path.exists(output_path) and errors_df.empty:
                if self.verbose:
                    print(f"Translations for {model_name} already exist. Skipping.")
                continue'''
                
            # Initialize translator
            if errors_df.empty:
                do_sample = False
                temperature = 0.1
            else:
                do_sample = True
                temperature = 1.0
            if "translator" in locals():
                translator.model.to("cpu")
                del translator
            
            torch.cuda.empty_cache()
            
            translator = Translator(
                model_name=model,
                temperature=temperature,
                do_sample=do_sample,
                device=self.device
            )
            
            # Initialize DataFrame to store results
            output_df = pd.DataFrame(columns=['audio_path', 'original_text', 'translated_text'])

            # Process each sample
            for i, (target_audio_path, _, target_transcription, _, _, _, _, _, _, _, _) in tqdm.tqdm(
                enumerate(self.dataloader), total=len(self.data)
            ):
                target_audio_path = target_audio_path[0]
                target_transcription = target_transcription[0]
                if i < self.skip:
                    continue
                if os.path.join('/home/jaime/datasets/SER/downloads/msppodcast/Audio', target_audio_path.split('/')[-1]) in prev_generated_data['audio_path'].values:
                    continue
                if not errors_df.empty:
                    if target_audio_path not in errors_df['audio_path'].values:
                        continue
                    
                if self.verbose:
                    print(f"Processing {target_audio_path}")
                    print(f"Original text: {target_transcription}")
                    
                # Translate the text
                translation = translator(
                    target_transcription,
                    source_language=self.source_language,
                    target_language=self.target_language,
                    
                )
                
                if self.verbose:
                    print(f"Translated text: {translation}")
                    
                # Store results
                new_row = {
                    'audio_path': target_audio_path,
                    'original_text': target_transcription,
                    'translated_text': translation,
                }
                output_df = pd.concat([output_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
            
            # Save translations
            output_df.to_csv(output_path, index=False)
            
            if self.verbose:
                print(f"Translation complete. Results saved to {output_path}")
                
    def evaluate_translations(self):
        """Evaluate translations and identify ones that need regeneration"""
        if self.verbose:
            print("Starting translation evaluation...")
            
        prev_generated_data = pd.read_csv(f'out/evaluation/translations/{self.source_database}_{self.target_language}.csv')

        errors_path = f"out/evaluation/translations/{self.source_database}_ERRORS_{self.target_language}.csv"
        if os.path.exists(errors_path):
            errors_df = pd.read_csv(errors_path)
        else:
            errors_df = pd.DataFrame()

        # Dictionary to track evaluation results by audio path
        all_translations = pd.DataFrame()
        gen_model_names = []

        # Evaluate translations from each model
        for gen_model in self.models:
            gen_model_name = gen_model.split('/')[-1]
            regen_translations_path = f"out/generation/translations/{self.source_database}_{gen_model_name}_{self.target_language}_regen.csv"
            if os.path.exists(regen_translations_path):
                translations_path = regen_translations_path
            else:
                translations_path = f"out/generation/translations/{self.source_database}_{gen_model_name}_{self.target_language}.csv"
            
            if "total_translations" not in locals():
                total_translations = len(pd.read_csv(f"out/generation/translations/{self.source_database}_{gen_model_name}_{self.target_language}.csv"))

            # Skip if translations don't exist
            if not os.path.exists(translations_path):
                if self.verbose:
                    print(f"Translations for {gen_model_name} not found. Skipping evaluation.")
                continue

            # Load translations
            translations_df = pd.read_csv(translations_path)
            
            if all_translations.empty:
                all_translations = translations_df
            else:
                all_translations = pd.merge(all_translations, translations_df[['audio_path', 'translated_text']], on='audio_path', how='inner')
            all_translations.rename(columns={'translated_text': gen_model_name}, inplace=True)
            
            gen_model_names.append(gen_model_name)

        eval_results = {}
        evaluation_path = f"out/evaluation/translations/{self.source_database}_{self.target_database}_{self.target_language}.csv"

        if not os.path.exists(evaluation_path) or not errors_df.empty: # TODO: Put True
            for eval_model in self.models:
                eval_model_name = eval_model.split('/')[-1]
                eval_results[eval_model_name] = {}

                # Initialize evaluator
                if "evaluator" in locals():
                    evaluator.model.to("cpu")
                    del evaluator
                torch.cuda.empty_cache()
                    
                evaluator = TranslationEvaluator(
                    model_name=eval_model,
                    device=self.device
                )
                
                # Evaluate each translation
                for _, row in tqdm.tqdm(all_translations.iterrows(), total=len(all_translations)):
                    
                    audio_path = row['audio_path']
                    eval_results[eval_model_name][audio_path] = {}
                    
                    if not errors_df.empty:
                        if audio_path not in errors_df['audio_path'].values:
                            continue
                        elif gen_model_name not in errors_df[errors_df['audio_path']==audio_path]['gen_model_name'].values:
                            continue
                    
                    original_text = row['original_text']

                    for gen_model_name in gen_model_names:
                        #if gen_model_name=='bloomz-7b1' and audio_path=="/home/jaime/datasets/SER/downloads/iemocap/Ses01F_impro04_F001.wav":
                        #    print("stop")
                        is_correct_item = evaluator.evaluate_translation(
                            source_text=original_text,
                            translation=row[gen_model_name],
                            source_lang=self.source_language,
                            target_lang=self.target_language
                        )
                        
                        eval_results[eval_model_name][audio_path][gen_model_name] = {
                            'translation': row[gen_model_name],
                            'is_correct': is_correct_item,
                        }
                        
            eval_models = list(eval_results.keys())

            # Intersección de audio_paths entre todos los evaluadores
            audio_paths = set.intersection(*(set(eval_results[em].keys()) for em in eval_models))

            data = []
            bad_data = []
            selected_audio_paths = []
            # Para cada audio común a todos los evaluadores
            for audio in audio_paths:
                # Intersección de modelos generadores para este audio
                gen_models = set.intersection(*(set(eval_results[em][audio].keys()) for em in eval_models))
                for gen_model in gen_models:
                    # Verificar que para todos los evaluadores is_correct sea True
                    if sum(eval_results[em][audio][gen_model]['is_correct'] for em in eval_models)==len(eval_models)-self.models_threshold:
                        if audio in selected_audio_paths:
                            continue
                        translation = eval_results[eval_models[0]][audio][gen_model]['translation']
                        
                        selected_audio_paths.append(audio)
                        data.append({
                            'audio_path': audio,
                            'gen_model_name': gen_model,
                            'translation': translation,
                            'is_correct': True
                        })
                    else:
                        if audio in selected_audio_paths:
                            continue
                        translation = eval_results[eval_models[0]][audio][gen_model]['translation']
                        bad_data.append({
                            'audio_path': audio,
                            'gen_model_name': gen_model,
                            'translation': translation,
                            'is_correct': False
                        })

            # Convertir la lista de registros en un DataFrame
            evaluation_df = pd.DataFrame(data)
            
            errors_df = pd.DataFrame(bad_data)
                
            
            if os.path.exists(evaluation_path):
                prev_evaluation_df = pd.read_csv(evaluation_path)
                evaluation_df = pd.concat([evaluation_df, prev_evaluation_df], ignore_index=True)
                print(f"Added {len(evaluation_df) - len(prev_evaluation_df)} new translations to {evaluation_path}. Remaining: {total_translations - len(evaluation_df)}")
                
            # combine evaluation_df with prev_generated_data
            
            evaluation_df.to_csv(evaluation_path, index=False)
            
            errors_df = errors_df[~errors_df['audio_path'].isin(evaluation_df['audio_path'])]
            
            errors_df.to_csv(errors_path, index=False)

            if self.verbose:
                print(f"Evaluation complete. Results saved to {evaluation_path}")
            
        if errors_df.empty:
            return False
        else:
            return True
                
    def run_pipeline(self):
        """Run the translation pipeline"""
        evaluate=True
        counter=0
        while evaluate:
            self.generate_translations()
            evaluate = self.evaluate_translations()
            counter+=1
            if counter>self.evaluation_attempts_until_threshold_decrease:
                self.models_threshold+=1
                counter=0
        

def main():
    parser = argparse.ArgumentParser(description="Run the translation pipeline.")
    
    parser.add_argument("--target_database", type=str, default='iemocap', help="Target database")
    parser.add_argument("--target_language", type=str, default="en", help="Target language (default: es)")
    parser.add_argument("--source_database", type=str, default='msppodcast', help="Source database")
    parser.add_argument("--source_language", type=str, default="en", help="Source language (default: en)")
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity (default: False)")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first n samples (default: 0)")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model on (default: cuda:0)")
    parser.add_argument("--fold", type=int, default=1, help="Fold to use for the dataset (default: 1)")
    parser.add_argument("--split", type=str, default="train", help="Split to use for the dataset (default: train)")
    parser.add_argument("--balance", action="store_true", help="Balance the dataset (default: False)")
    parser.add_argument("--sample", action="store_true", help="Sample the dataset (default: False)")
    parser.add_argument("--force_balance", action="store_true", help="Force balance the dataset (default: False)")
    parser.add_argument("--evaluation_attempts_until_threshold_decrease", type=int, default=3, help="Number of evaluation attempts until threshold decrease (default: 3)")
    parser.add_argument("--increase_max_count", type=float, default=1.2, help="Increase max count for balance (default: 1)")
    
    args = parser.parse_args()
    
    pipeline = TranslationPipeline(
        target_database=args.target_database,
        target_language=args.target_language,
        source_database=args.source_database,
        source_language=args.source_language,
        device=args.device,
        verbose=args.verbose,
        skip=args.skip,
        fold=args.fold,
        split=args.split,
        balance=True,
        sample=False,
        force_balance=False,
        evaluation_attempts_until_threshold_decrease=args.evaluation_attempts_until_threshold_decrease,
        increase_max_count=args.increase_max_count,
    )
    
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
