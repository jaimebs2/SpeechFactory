import os
import torch
import torch.nn as nn
import argparse
import sys
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from sklearn.metrics import f1_score
from scipy.stats import entropy
import argparse

sys.path.append(".")
from src.evaluator.whisper_ser.classificator import Classificator
from src.evaluator.whisper_ser.processor import WhisperPreprocess
from src.evaluator.whisper_ser.config import WhisperSERConfig
from src.evaluator.whisper_ser.pooling import AttentionMeanPooling

class EmotionClassifier(nn.Module):
    def __init__(self, config=WhisperSERConfig(), dtype=torch.bfloat16, device="cuda", model_path="models_weights/whisper-ser-iemocap/whisper-ser-iemocap.ckpt"):
        """
        Initialize the emotion recognition model with WhisperSER
        
        Args:
            model_path (str, optional): Path to the pretrained model weights
        """
        super(EmotionClassifier, self).__init__()
        # Initialize the model and processor
        self.model = Classificator(config)
        self.preprocessor = WhisperPreprocess()
        
        # Load model weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded model weights from {model_path}")
        
        # Set to evaluation mode
        self.model.eval()
        self.model.to(device=device, dtype=dtype)
        self.softmax = nn.Softmax(dim=1).to(device=device, dtype=dtype)
        self.device =device
        self.dtype = dtype
        
    def forward(self, audio, sr):
        """
        Predict the emotion from an audio file
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            tuple: (emotion, probabilities)
        """
        # Preprocess the audio using the WhisperPreprocess
        inputs = self.preprocessor(audio, sr)
        
        # Move tensors to the appropriate device
        audio_features = inputs['input_features'].to(self.device, dtype=self.dtype)

        audio_len = torch.tensor([len(audio) / sr]).to(self.device)
        
        # Make prediction
        emotion, probabilities, features = self.model.predict(audio_features, audio_len)
        
        probabilities = self.softmax(probabilities)
        
        return emotion, probabilities.to("cpu", float).squeeze(0).numpy(), features.to("cpu")

def main():
    parser = argparse.ArgumentParser(description="Evaluate emotion recognition on TTS-generated audio files.")

    parser.add_argument("--audio_folder", type=str, default="out/generation/tts/iemocap_fishspeech/es", help="Folder containing TTS-generated .wav files")
    parser.add_argument("--ground_truth_csv", type=str, default="out/iemocap.csv", help="CSV file with ground truth emotions")
    parser.add_argument("--output_path", type=str, default="out/evaluation/emotions/emotion_predictions.csv", help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model on (default: cuda:0)")
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity (default: False)")
    
    args = parser.parse_args()
    
    audio_folder = args.audio_folder
    ground_truth_csv = args.ground_truth_csv
    output_path = args.output_path
    device = args.device
    dtype = torch.bfloat16
    verbose = args.verbose
    
    # Initialize output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize the models
    model_iemocap = EmotionClassifier(
        config=WhisperSERConfig(
            projector_layers=[1280, 4],
            batch_norm=True,
            pooling=AttentionMeanPooling,
            embed_dim=1280,
            emo_dict={0: 'neutral', 1: 'happy', 2: 'angry', 3: 'sad'},
            activation=torch.nn.GELU
        ),
        device=device,
        dtype=dtype, 
        model_path="models_weights/whisper-ser-iemocap/whisper-ser-iemocap.ckpt"
    )
    model_meacorpus = EmotionClassifier(
        config=WhisperSERConfig(
            projector_layers=[1280, 6],
            batch_norm=True,
            pooling=AttentionMeanPooling,
            embed_dim=1280,
            emo_dict={0: 'neutral', 1: 'disgust', 2: 'happy', 3: 'fear', 4: 'angry', 5: 'sad'}, # processed/emotion_tokens
            activation=torch.nn.GELU
        ),
        device=device,
        dtype=dtype,
        model_path="models_weights/whisper-ser-meacorpus/whisper-ser-meacorpus_fold-1.ckpt"
    )
    
    # Load ground truth data
    ground_truth_df = pd.read_csv(ground_truth_csv)
    
    # Initialize DataFrame to store results
    results_df = pd.DataFrame(columns=[
        'audio_path', 
        'real_emotion', 
        'iemocap_emotion', 'iemocap_max_prob', 'iemocap_entropy',
        'meacorpus_emotion', 'meacorpus_max_prob', 'meacorpus_entropy'
    ])
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
    
    # Process each audio file
    for audio_file in tqdm.tqdm(audio_files, total=len(audio_files)):
        audio_path = os.path.join(audio_folder, audio_file)
        
        # Extract base filename without extension to match with ground truth
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Find corresponding ground truth
        ground_truth_row = ground_truth_df[ground_truth_df['audio_path'].str.contains(base_filename)]
        
        if len(ground_truth_row) == 0:
            if verbose:
                print(f"No ground truth found for {base_filename}. Skipping.")
            continue
            
        real_emotion = ground_truth_row['real_voice_emotion'].values[0]
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        audio = audio.squeeze(0)
        
        # Get predictions from both models
        iemocap_emotion, probs = model_iemocap(audio.to(device=device), sr)
        
        iemocap_probs = {
            "neutral": probs[0].item(),
            "happy": probs[1].item(),
            "angry": probs[2].item(),
            "sad": probs[3].item()
        }
        
        meacorpus_emotion, probs = model_meacorpus(audio.to(device=device), sr)
        
        meacorpus_probs = {
            "neutral": probs[0].item(),
            "sad": probs[1].item(), #disgust
            "happy": probs[2].item(),
            "fear": probs[3].item(),
            "angry": probs[4].item(),
            "sad": probs[5].item()
        }
        
        # Calculate entropy and max probability for each model
        iemocap_max_prob = max(iemocap_probs.values())
        meacorpus_max_prob = max(meacorpus_probs.values())
        
        # Calculate entropy (higher entropy means more uncertainty)
        iemocap_entropy = entropy(list(iemocap_probs.values()))
        meacorpus_entropy = entropy(list(meacorpus_probs.values()))
        
        if verbose:
            print(f"Audio: {audio_path}")
            print(f"Real emotion: {real_emotion}")
            print(f"IEMOCAP prediction: {iemocap_emotion} (prob: {iemocap_max_prob:.4f}, entropy: {iemocap_entropy:.4f})")
            print(f"MEA-Corpus prediction: {meacorpus_emotion} (prob: {meacorpus_max_prob:.4f}, entropy: {meacorpus_entropy:.4f})")
            print("---")
        
        # Store results
        new_row = {
            'audio_path': audio_path,
            'real_emotion': real_emotion,
            'iemocap_emotion': iemocap_emotion,
            'iemocap_max_prob': iemocap_max_prob,
            'iemocap_entropy': iemocap_entropy,
            'meacorpus_emotion': meacorpus_emotion,
            'meacorpus_max_prob': meacorpus_max_prob,
            'meacorpus_entropy': meacorpus_entropy
        }
        
        results_df = pd.concat([results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    
    # Save results to CSV
    results_df.to_csv(output_path, index=False)
    
    # Calculate F1 scores for each model
    # Convert string labels to numeric for F1 calculation
    iemocap_emotion_mapping = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3}
    meacorpus_emotion_mapping = {'neutral': 0, 'disgust': 1, 'happy': 2, 'fear': 3, 'angry': 4, 'sad':5}
    
    y_true = [iemocap_emotion_mapping.get(emotion, -1) for emotion in results_df['real_emotion']]
    y_pred_iemocap = [iemocap_emotion_mapping.get(emotion, -1) for emotion in results_df['iemocap_emotion']]
    y_pred_meacorpus = [meacorpus_emotion_mapping.get(emotion, -1) for emotion in results_df['meacorpus_emotion']]
    
    # Filter out any entries where mapping failed
    valid_indices = [i for i, val in enumerate(y_true) if val != -1 and y_pred_iemocap[i] != -1 and y_pred_meacorpus[i] != -1]
    
    y_true_filtered = [y_true[i] for i in valid_indices]
    y_pred_iemocap_filtered = [y_pred_iemocap[i] for i in valid_indices]
    y_pred_meacorpus_filtered = [y_pred_meacorpus[i] for i in valid_indices]
    
    # Calculate F1 scores for each model
    iemocap_f1_macro = f1_score(y_true_filtered, y_pred_iemocap_filtered, average='macro')
    iemocap_f1_weighted = f1_score(y_true_filtered, y_pred_iemocap_filtered, average='weighted')
    
    meacorpus_f1_macro = f1_score(y_true_filtered, y_pred_meacorpus_filtered, average='macro')
    meacorpus_f1_weighted = f1_score(y_true_filtered, y_pred_meacorpus_filtered, average='weighted')
    
    # Print F1 scores
    print("\nF1 Scores:")
    print(f"IEMOCAP Model - Macro F1: {iemocap_f1_macro:.4f}, Weighted F1: {iemocap_f1_weighted:.4f}")
    print(f"MEA-Corpus Model - Macro F1: {meacorpus_f1_macro:.4f}, Weighted F1: {meacorpus_f1_weighted:.4f}")
    
    # Save F1 scores to a separate CSV
    metrics_df = pd.DataFrame({
        'model': ['iemocap', 'meacorpus'],
        'f1_macro': [iemocap_f1_macro, meacorpus_f1_macro],
        'f1_weighted': [iemocap_f1_weighted, meacorpus_f1_weighted]
    })
    
    metrics_path = os.path.splitext(output_path)[0] + "_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    if verbose:
        print(f"\nResults saved to {output_path}")
        print(f"Metrics saved to {metrics_path}")
        
        
    from sklearn.metrics import roc_curve, auc
    def compute_roc(y_true, scores):
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc
    
    for model in ["iemocap", "meacorpus"]:
        results_df[f"{model}_correct"] = results_df.apply(lambda x: x['real_emotion'] == x[f"{model}_emotion"], axis=1)
    
    fpr_iemocap, tpr_iemocap, thr_iemocap, roc_auc_iemocap = compute_roc(results_df['iemocap_correct'], results_df['iemocap_max_prob'].values)
    fpr_meacorpus, tpr_meacorpus, thr_meacorpus, roc_auc_meacorpus = compute_roc(results_df['meacorpus_correct'], results_df['meacorpus_max_prob'].values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_iemocap, tpr_iemocap, label=f'IEMOCAP (AUC = {roc_auc_iemocap:.2f})')
    plt.plot(fpr_meacorpus, tpr_meacorpus, label=f'MEA-Corpus (AUC = {roc_auc_meacorpus:.2f})')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Emotion Recognition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.splitext(output_path)[0] + "_roc.png")
    
    def select_threshold(fpr, tpr, thresholds, target_fpr=0.02):
        idx = np.where(fpr <= target_fpr)[0][-1]
        return thresholds[idx]
    
    iemocap_threshold = select_threshold(fpr_iemocap, tpr_iemocap, thr_iemocap)
    meacorpus_threshold = select_threshold(fpr_meacorpus, tpr_meacorpus, thr_meacorpus)
    
    # save thresholds
    thresholds_df = pd.DataFrame({
        'model': ['iemocap', 'meacorpus'],
        'threshold': [iemocap_threshold, meacorpus_threshold]
    })
    
    thresholds_path = os.path.splitext(output_path)[0] + "_thresholds.csv"
    thresholds_df.to_csv(thresholds_path, index=False)
    

if __name__ == "__main__":
    main()
