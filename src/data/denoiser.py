import torchaudio
from torch import nn
from speechbrain.inference import SepformerSeparation

class Denoiser(nn.Module):
    def __init__(self, model_path="speechbrain/sepformer-wham16k-enhancement", 
                 savedir='models_weights/sepformer-wham16k-enhancement'):
        """
        Initialize the audio denoiser with the Sepformer model.
        
        Args:
            model_path (str): Path or name of the pretrained Sepformer model.
            savedir (str): Directory to save the pretrained model.
        """
        super(Denoiser, self).__init__()
        self.model = SepformerSeparation.from_hparams(
            source=model_path, 
            savedir=savedir
        )
        
    def __call__(self, audio_tensor, input_sr, output_path=None, output_noise=False):
        """
        Denoise an audio tensor and optionally save the result.
        
        Args:
            audio_tensor (torch.Tensor): Input audio tensor of shape [channels, samples].
            input_sr (int): Sampling rate of the input audio tensor.
            output_path (str, optional): Path to save the denoised audio.
            output_noise (bool, optional): If True, save the noise audio instead of the denoised audio.
            
        Returns:
            torch.Tensor: Denoised audio tensor (single channel).
        """
        # Move tensor to the same device as the model
        audio_tensor = audio_tensor.to(self.model.device)
        
        # If the input has multiple channels, convert to mono by averaging
        if audio_tensor.size(0) > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
            
        # Resample the audio if its sampling rate doesn't match the model's expected rate
        fs_model = self.model.hparams.sample_rate
        if input_sr != fs_model:
            print(f"Resampling the audio from {input_sr} Hz to {fs_model} Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=input_sr, new_freq=fs_model).to(self.model.device)
            audio_tensor = resampler(audio_tensor)
        
        # Run the separation model on the audio tensor
        self.model.to("cpu")
        enhanced_sources = self.model.separate_batch(audio_tensor)
        # Normalize the output similar to what separate_file does
        enhanced_sources = enhanced_sources / enhanced_sources.abs().max(dim=1, keepdim=True)[0]
        # Extract the denoised audio (first source channel)
        denoised_audio = enhanced_sources[:, :, 0].detach().cpu()
        
        # Resample the denoised audio back to the original sampling rate
        if input_sr != fs_model:
            print(f"Resampling the denoised audio from {fs_model} Hz to {input_sr} Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=fs_model, new_freq=input_sr)
            denoised_audio = resampler(denoised_audio)
            
        if output_noise:
            noise = enhanced_sources[:, :, 1].detach().cpu()
            if input_sr != fs_model:
                print(f"Resampling the noise audio from {fs_model} Hz to {input_sr} Hz")
                noise = resampler(noise)

        # Optionally save the denoised audio using the target sample rate
        if output_path:
            torchaudio.save(output_path, denoised_audio, input_sr)
            if output_noise:
                torchaudio.save(output_path.replace('.wav', '_noise.wav'), noise, input_sr)
    
        if output_noise:
            return (denoised_audio, noise)
        else:        
            return denoised_audio
    