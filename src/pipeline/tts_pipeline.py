# transformers==4.36.0
import argparse
import os
import torch
import csv
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from torch.utils.data import DataLoader
from jiwer import wer
import random
import torchaudio
import configparser
from scipy.stats import entropy
from sklearn.metrics import f1_score

#In this pipeline, the generated audio needs to have a WER less than a threshold to be saved. Add an other filter to save the audio. In this case, 

import sys
sys.path.append(".")
from src.generator.fishspeechwrapper import FishSpeechWrapper
from src.data.data_module import EmoDataset
from src.evaluator.asr import ASR
from src.evaluator.embedders import GlobalCosineSimilarity

from src.evaluator.ser import AttentionMeanPooling, WhisperSERConfig, EmotionClassifier

main_config = configparser.ConfigParser()
main_config.read('config/config.ini')

class TTSPipeline:
    def __init__(self, target_database, target_language, source_database, source_language, 
                 device="cuda:0", verbose=False, skip=0, limit=None, temperature=0.7, 
                 top_p=0.7, denoise=True, evaluate=True, evaluation_only=False,
                 max_regenerations=5, wer_threshold=1.0, cosine_similarity_threshold=0.5, fold=1, split="valid", embedder_model="hubert", rebuilt=False, balance=False, sample=False, force_balance=True, increase_max_count=1.0):
        self.target_database = target_database
        self.target_language = target_language
        self.source_database = source_database
        self.source_language = source_language
        self.device = device
        self.verbose = verbose
        self.skip = skip
        self.limit = limit
        self.temperature = temperature
        self.top_p = top_p
        self.denoise = denoise
        self.evaluate = evaluate
        self.evaluation_only = evaluation_only
        self.max_regenerations = max_regenerations
        self.wer_threshold = wer_threshold
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.fold = fold
        self.split = split
        self.embedder_model = embedder_model
        self.balance = balance
        self.sample = sample
        self.force_balance = force_balance
        self.rebuilt = rebuilt
        self.increase_max_count = increase_max_count
        
        # Setup output directory
        self.save_dir = os.path.join('out/generation/tts', f'{target_database}_fishspeech', f'{source_language}_{source_database}')
        self.base_save_dir = os.path.join('out/evaluation/tts', f'{target_database}_fishspeech', f'{source_language}_{source_database}', 'source')
        self.target_save_dir = os.path.join('out/evaluation/tts', f'{target_database}_fishspeech', f'{source_language}_{source_database}', 'target')
        self.eval_save_dir = os.path.join('out/evaluation/tts', f'{target_database}_fishspeech', f'{source_language}_{source_database}')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.base_save_dir, exist_ok=True)
        os.makedirs(self.target_save_dir, exist_ok=True)
        
        if rebuilt:
            for file in os.listdir(self.base_save_dir):
                if file.endswith('.wav'):
                    os.remove(os.path.join(self.base_save_dir, file))
            for file in os.listdir(self.target_save_dir):
                if file.endswith('.wav'):
                    os.remove(os.path.join(self.target_save_dir, file))
                    
            for file in os.listdir(self.save_dir):
                if file.endswith('.wav'):
                    os.remove(os.path.join(self.save_dir, file))
                
            

        # Translations file path
        self.translations_path = os.path.join('out/evaluation/translations', 
                                             f'{target_database}_{source_database}_{source_language}.csv')
        
        # Initialize models
        self._init_models()
        
        # Initialize dataset only if not in evaluation-only mode
        if not self.evaluation_only:
            self._init_dataset()
    
    def _init_models(self):
        """Initialize the FishSpeech model and ASR evaluator"""
        if self.verbose:
            print("Loading models...")
        
        self.device = torch.device(self.device)
        
        # Initialize models selectively based on task
        if not self.evaluation_only:
            # Default model paths - can be made configurable if needed
            vocoder_path = "models_weights/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
            llm_path = "models_weights/fish-speech-1.5"
            
            # Initialize the FishSpeech wrapper
            self.fish_model = FishSpeechWrapper(vocoder_path, self.device)
            self.fish_model.load_text_to_semantic_model(llm_path)
        
        # Initialize ASR model for evaluation
        if self.evaluate:
            if self.verbose:
                print("Loading ASR model for evaluation...")
            self.asr_model = ASR(self.device)
            self.hubert_cosine_similarity = GlobalCosineSimilarity(model_type=self.embedder_model, device="cpu", use_processor=True)
    
    def _init_dataset(self):
        """Initialize the dataset for voice references"""
        if self.verbose:
            print(f"Loading {self.source_database} dataset")
        
        self.data = EmoDataset(
            target=self.target_database,
            reference=self.source_database,
            fold=self.fold,
            language_code=self.source_language,
            device=self.device,
            denoise=self.denoise,
            skip=self.skip,
            audio_description=False,
            split=self.split,
            balance=self.balance,
            sample=self.sample,
            force_balance=self.force_balance,
            increase_max_count=self.increase_max_count,
        )
        
        self.dataloader = DataLoader(self.data, batch_size=1, shuffle=False)
    
    def evaluate_audio(self, audio_path, reference_text):
        """
        Evaluate generated audio using ASR and calculate WER
        
        Args:
            audio_path (str): Path to the generated audio file
            reference_text (str): Reference text (expected transcription)
            
        Returns:
            dict: Dictionary with ASR transcription and WER
        """
        # Load audio file
        audio, sr = sf.read(audio_path)
        
        # Get ASR transcription
        
        # turn audio into 16000
        if sr != 16000:
            if not hasattr(self, 'resampler'):
                self.resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = self.resampler(torch.tensor(audio).unsqueeze(0).to(dtype=torch.float)).squeeze().numpy()
            sr = 16000
        asr_transcription = self.asr_model.transcribe(audio)
        
        # Calculate WER
        error_rate = wer(reference_text, asr_transcription)
        
        return {
            "asr_transcription": asr_transcription,
            "reference_text": reference_text,
            "wer": error_rate
        }
    
    def get_new_reference_voice(self, emotion, seed):
        """Get a new reference voice for the given emotion with a different seed"""
        new_seed = seed + random.randint(100, 10000)  # Use a different seed
        new_ref_path = self.data.reference_data[self.data.reference_data['emotion'] == emotion]['audio_path'].sample(1, random_state=new_seed).values[0]
        new_ref_path = os.path.join(main_config['DATA']['FILES_PATH'], new_ref_path)
        
        # Load the new reference audio
        new_ref_audio, new_ref_sr = torchaudio.load(new_ref_path)
        
        # Apply denoising if enabled
        if self.denoise and hasattr(self.data, 'denoiser'):
            if new_ref_sr != 16000:
                if not hasattr(self.data, 'resampler') or self.data.resampler is None:
                    self.data.resampler = torchaudio.transforms.Resample(orig_freq=new_ref_sr, new_freq=16000)
                new_ref_audio = self.data.resampler(new_ref_audio)
                new_ref_sr = 16000
            new_ref_audio = self.data.denoiser(new_ref_audio, new_ref_sr, output_path=None, output_noise=False)
        
        # Apply trimming effects
        if hasattr(self.data, 'effects'):
            trimmed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                new_ref_audio, new_ref_sr, self.data.effects
            )
            if trimmed_audio.shape[1] != 0:
                new_ref_audio = trimmed_audio
        
        return new_ref_path, new_ref_audio, new_ref_sr
    
    def run_pipeline(self):
        """Run the TTS pipeline to generate synthetic voices and then evaluate them"""
        # Check if translations file exists
        if not os.path.exists(self.translations_path):
            raise FileNotFoundError(f"Translations file not found: {self.translations_path}")
        
        # Load translations
        translations_df = pd.read_csv(self.translations_path)
        
        if not self.rebuilt:
            existing_tts_files = os.listdir(self.save_dir)
        else:
            existing_tts_files = []
        
        # Skip generation if all files exist or in evaluation-only mode
        if self.evaluation_only:
                print(f"Skipping generation phase, running evaluation only.")
        
        # Only run generation if needed
        else:
            # Process each sample - generation phase
            total_samples = min(len(self.data), self.limit) if self.limit else len(self.data)
            print(f"Processing {total_samples} samples")
            
            regeneration_stats = {"total_samples": 0, "regenerated": 0, "attempts": [], "skipped_audios": 0}
            
            emotions_df = pd.DataFrame()
            for i, (target_audio_path, target_audio, target_transcription, target_sr, 
                    reference_audio_path, reference_audio, reference_sr, 
                    reference_transcription, pitch_std, speaking_rate, emotion) in enumerate(
                    tqdm(self.dataloader, total=total_samples, desc="Generating")):
                        
                if i < self.skip:
                    continue
                
                if not self.rebuilt and target_audio_path[0].split('/')[-1] in existing_tts_files:
                    continue
                    
                if self.limit is not None and i >= self.skip + self.limit:
                    break
                
                translation = translations_df[translations_df['audio_path'] == target_audio_path[0]]['translation'].values[0]
                
                # Get the filename from the path
                target_filename = target_audio_path[0].split("/")[-1]
                out_path = os.path.join(self.save_dir, target_filename)
                
                if os.path.exists(out_path):
                    if self.verbose:
                        print(f"Skipping existing file: {out_path}")
                    continue
                
                regeneration_stats["total_samples"] += 1
                
                if self.verbose:
                    print(f"Processing sample {i+1}/{total_samples}: {target_filename}")
                    print(f"  Reference: {reference_transcription[0]}")
                    print(f"  Target text: {translation}")
                    print(f"  Emotion: {emotion[0]}")
                
                # Try generating with quality check
                success = False
                attempts = 0
                current_ref_audio = reference_audio
                current_ref_sr = reference_sr
                current_ref_path = reference_audio_path[0]
                
                while not success and attempts < self.max_regenerations:
                    try:
                        # Generate audio with current reference
                        fake_audio, sample_rate = self.fish_model.clone_voice(
                            reference_audio=current_ref_audio[0] if attempts == 0 else current_ref_audio,
                            reference_sr=current_ref_sr,
                            text=translation,
                            temperature=self.temperature,
                            top_p=self.top_p
                        )
                        
                        # Immediately check quality with ASR if evaluation is enabled
                        if self.evaluate:
                            # Save to a temporary file for evaluation
                            temp_path = out_path + ".temp.wav"
                            sf.write(temp_path, fake_audio, sample_rate)
                            
                            # Evaluate with ASR
                            eval_result = self.evaluate_audio(temp_path, translation)
                            current_wer = eval_result["wer"]
                            
                            if self.verbose:
                                print(f"  Generated WER: {current_wer:.4f} (Attempt {attempts+1})")
                            
                            if current_wer < self.wer_threshold:
                                # First filter passed.
                                cosine_similarity = self.hubert_cosine_similarity(current_ref_path, temp_path)
                                
                                if self.verbose:
                                    print(f"  Hubert cosine similarity: {cosine_similarity:.4f}")
                                
                                if cosine_similarity > self.cosine_similarity_threshold:
                                    # Second filter passed.       
                                    os.rename(temp_path, out_path)
                                    success = True
                                    if self.verbose:
                                        print(f"  Quality acceptable! Saved to: {out_path}")
                                        
                                    # save current_ref_path audio to base_out_path
                                    source_out_path = os.path.join(self.base_save_dir, target_filename)
                                    target_out_path = os.path.join(self.target_save_dir, target_filename)
                                    current_ref_audio = current_ref_audio[0] if attempts == 0 else current_ref_audio
                                    target_audio = target_audio[0].mean(dim=0, keepdim=True)
                                    torchaudio.save(source_out_path, current_ref_audio.mean(dim=0, keepdim=True), current_ref_sr)
                                    torchaudio.save(target_out_path, target_audio, target_sr)
                                    sf.write(out_path, fake_audio, sample_rate)
                                    new_row = pd.DataFrame([{'audio_path': target_filename, 'emotion': emotion[0]}])
                                    emotions_df = pd.concat([emotions_df, new_row], ignore_index=True)
                            
                            if not success:
                                # Bad quality, delete temp file and try a new voice
                                os.remove(temp_path)
                                attempts += 1
                                regeneration_stats["regenerated"] += 1
                                
                                if attempts < self.max_regenerations:
                                    if self.verbose:
                                        print(f"  Quality too low (WER={current_wer:.4f}). Trying new voice...")
                                    
                                    # Get a new reference voice for the same emotion
                                    current_ref_path, current_ref_audio, current_ref_sr = self.get_new_reference_voice(
                                        emotion[0], i + self.data.seed + attempts
                                    )
                                    
                                    if self.verbose:
                                        print(f"  New reference: {current_ref_path}")
                                else:
                                    # Reached max attempts, save the best one we got
                                    if self.verbose:
                                        print(f"  Reached max regenerations. Skipping audio.")
                                        
                                    regeneration_stats['skipped_audios'] += 1
                                    new_row = pd.DataFrame([{'audio_path': target_filename, 'emotion': emotion[0]}])
                                    emotions_df = pd.concat([emotions_df, new_row], ignore_index=True)
    
                        else:
                            # No evaluation, just save
                            sf.write(out_path, fake_audio, sample_rate)
                            success = True
                            if self.verbose:
                                print(f"  Saved to: {out_path}")
                        
                    except Exception as e:
                        print(f"Error processing {target_audio_path[0]}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        attempts += 1
                        if attempts >= self.max_regenerations:
                            break
                
                regeneration_stats["attempts"].append(attempts + 1)  # +1 because first attempt counts
            
            # Print regeneration statistics
            if regeneration_stats["total_samples"] > 0:
                avg_attempts = sum(regeneration_stats["attempts"]) / len(regeneration_stats["attempts"])
                print(f"Generation completed. Generated files saved to {self.save_dir}")
                print(f"Regeneration statistics:")
                print(f"  Total samples: {regeneration_stats['total_samples']}")
                print(f"  Samples requiring regeneration: {regeneration_stats['regenerated']} ({regeneration_stats['regenerated']/regeneration_stats['total_samples']*100:.1f}%)")
                print(f"  Skipped audios: {regeneration_stats['skipped_audios']}")
                print(f"  Average attempts per sample: {avg_attempts:.2f}")
        
                # delete all models
                del self.fish_model
                del self.asr_model
                del self.hubert_cosine_similarity
                torch.cuda.empty_cache()
                
                ser_paths = ["models_weights/whisper-ser-iemocap/whisper-ser-iemocap.ckpt", 
                             "models_weights/whisper-ser-iemocap_meacorpus/whisper-ser-iemocap_meacorpus.ckpt",
                             "models_weights/whisper-ser-meacorpus/whisper-ser-meacorpus.ckpt"]
                
                fieldnames = [
                        "fold",
                        "wer_threshold",
                        "embedder_model",
                        "cosine_similarity_threshold",
                        "ser_name",
                        "max_regenerations",
                        "temperature",
                        "top_p",
                        "denoise",
                        "skipped_audios",
                        "avg_attempts",
                        "total_samples",
                        "f1_macro"
                    ]
                
                for path in ser_paths:
                    if "ser_model" in locals():
                        del ser_model
                        torch.cuda.empty_cache()
                        
                    # Init SER model
                    ser_model = EmotionClassifier(
                        config=WhisperSERConfig(
                            projector_layers=[1280, 4],
                            batch_norm=True,
                            pooling=AttentionMeanPooling,
                            embed_dim=1280,
                            emo_dict={0: 'neutral', 1: 'happy', 2: 'angry', 3: 'sad'},
                            activation=torch.nn.GELU
                        ),
                        device=self.device,
                        dtype=torch.bfloat16, 
                        model_path=path
                    )
                    
                    # Evaluate SER for all files in self.save_dir
                    
                    predicted_emotion_list=[]
                    neutral_list=[]
                    happy_list=[]
                    angry_list=[]
                    sad_list=[]
                    entropy_score_list=[]

                    for audio_file in tqdm(emotions_df['audio_path'], desc="Evaluating SER"):
                        if os.path.exists(os.path.join(self.save_dir, audio_file)):
                            audio, sr = torchaudio.load(os.path.join(self.save_dir, audio_file))
                            emotion, probs, _ = ser_model(audio.to(device=self.device), sr)
                            probs_dict = {
                                "neutral": probs[0].item(),
                                "happy": probs[1].item(),
                                "angry": probs[2].item(),
                                "sad": probs[3].item()
                            }
                            entropy_score = entropy(list(probs_dict.values()))
                        else:
                            emotion = None
                            probs_dict = {
                                "neutral": None,
                                "happy": None,
                                "angry": None,
                                "sad": None
                            }
                            entropy_score = None
                        
                        
                        predicted_emotion_list.append(emotion)
                        neutral_list.append(probs_dict["neutral"])
                        happy_list.append(probs_dict["happy"])
                        angry_list.append(probs_dict["angry"])
                        sad_list.append(probs_dict["sad"])

                        entropy_score_list.append(entropy_score)
                        
                    emotions_df['predicted_emotion'] = predicted_emotion_list
                    emotions_df['neutral'] = neutral_list
                    emotions_df['happy'] = happy_list
                    emotions_df['angry'] = angry_list
                    emotions_df['sad'] = sad_list
                    emotions_df['entropy'] = entropy_score_list
                    
                    if self.verbose:
                        print(f"SER evaluation completed. Results saved to {self.eval_save_dir}")
                        
                    # Obtain f1 macro score
                    # lower strings
                    name = path.split("-")[-1][:-5]
                    emotions_df['emotion'] = emotions_df['emotion'].str.lower()
                    emotions_df.to_csv(os.path.join(self.eval_save_dir, f"final-emotions_wer-{self.wer_threshold}_{self.embedder_model}-{self.cosine_similarity_threshold}-{name}_fold-{self.fold}.csv"), index=False)
                    emotions_df = emotions_df.dropna(subset=['predicted_emotion']).reset_index(drop=True)
                    f1_macro = f1_score(emotions_df['emotion'], emotions_df['predicted_emotion'], average='macro')
                    print(f"F1 macro score: {f1_macro:.4f}")
                    
                    csv_file = os.path.join(self.eval_save_dir, "f1_macro.csv")
                    if not os.path.exists(csv_file):
                        with open(csv_file, mode="w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            
                    with open(csv_file, mode="a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow({
                            "fold": self.fold,
                            "wer_threshold": self.wer_threshold,
                            "embedder_model": self.embedder_model,
                            "cosine_similarity_threshold": self.cosine_similarity_threshold,
                            "ser_name": name,
                            "max_regenerations": self.max_regenerations,
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "denoise": self.denoise,
                            "skipped_audios": regeneration_stats['skipped_audios'],
                            "avg_attempts": avg_attempts,
                            "total_samples": regeneration_stats['total_samples'],
                            "f1_macro": f1_macro
                        })


def main():
    parser = argparse.ArgumentParser(description="Run the tts pipeline.")
    
    parser.add_argument("--target_database", type=str, default='msppodcast', help="Target database")
    parser.add_argument("--target_language", type=str, default="en", help="Target language (default: en)")
    parser.add_argument("--source_database", type=str, default='iemocap', help="Source database")
    parser.add_argument("--source_language", type=str, default="en", help="Source language (default: es)")
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity (default: False)")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first n samples (default: 0)")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model on (default: cuda:0)")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first n samples after skip")
    parser.add_argument("--temperature", type=float, default=1.5, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p for sampling")
    parser.add_argument("--denoise", type=bool, default=False, help="Denoise the audio")
    parser.add_argument("--evaluate", type=bool, default=True, help="Evaluate generated audio with ASR")
    parser.add_argument("--fold", type=int, default=1, help="Fold number for SER evaluation")
    parser.add_argument("--split", type=str, default="train", help="Split for SER evaluation")
    parser.add_argument("--evaluation_only", action="store_true", 
                       help="Skip generation and only run evaluation on existing files")
    
    parser.add_argument("--max_regenerations", type=int, default=10, 
                        help="Maximum number of regeneration attempts if quality is low")
    parser.add_argument("--wer_threshold", type=float, default=0.3, 
                        help="WER threshold below which generation is acceptable")
    parser.add_argument("--cosine_similarity_threshold", type=float, default=0.7,
                        help="Hubert cosine similarity threshold below which generation is acceptable")
    
    parser.add_argument("--embedder_model", type=str, default="hubert", help="wav2vec2, hubert, xvector, ecapa")
    
    parser.add_argument("--rebuilt", type=bool, default=False, help="Rebuilt the wavs")
    
    parser.add_argument("--balance", action="store_true", help="Balance the dataset (default: False)")
    parser.add_argument("--sample", action="store_true", help="Sample the dataset (default: False)")
    parser.add_argument("--force_balance", action="store_true", help="Force balance the dataset (default: False)")
    parser.add_argument("--increase_max_count", type=float, default=1.2, help="Increase max count for balance (default: 1.0)")
    
    args = parser.parse_args()
    
    # Ensure evaluate is True if evaluation_only is True
    if args.evaluation_only:
        args.evaluate = True
    
    pipeline = TTSPipeline(
        target_database=args.target_database,
        target_language=args.target_language,
        source_database=args.source_database,
        source_language=args.source_language,
        device=args.device,
        verbose=args.verbose,
        skip=args.skip,
        limit=args.limit,
        temperature=args.temperature,
        top_p=args.top_p,
        denoise=args.denoise,
        evaluate=args.evaluate,
        evaluation_only=args.evaluation_only,
        max_regenerations=args.max_regenerations,
        wer_threshold=args.wer_threshold,
        cosine_similarity_threshold=args.cosine_similarity_threshold,
        fold=args.fold,
        split=args.split,
        embedder_model=args.embedder_model,
        rebuilt=args.rebuilt,
        balance=True, #not sure if this is necessary
        sample=False, #not sure if this is necessary
        force_balance=False, #not sure if this is necessary
        increase_max_count=args.increase_max_count
    )
    
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
