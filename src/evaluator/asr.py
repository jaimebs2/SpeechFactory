import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class ASR:
    def __init__(self, device="cuda", model_id="openai/whisper-large-v3"):
        self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
    
    def transcribe(self, audio_tensor, return_timestamps=False):
        """
        Transcribe the given audio tensor using the ASR model.
        
        Args:
            audio_tensor: The audio tensor or file path to transcribe
            return_timestamps (bool): Whether to return timing information
            
        Returns:
            str or dict: The transcribed text or full result with timestamps
        """
        # Convert PyTorch tensor to numpy array if necessary
        if isinstance(audio_tensor, torch.Tensor):
            audio_tensor = audio_tensor.detach().cpu().numpy()
            
        result = self.pipe(audio_tensor, return_timestamps=return_timestamps)
        if not return_timestamps:
            return result["text"]
        return result


# Example usage
if __name__ == "__main__":
    asr = ASR()
    audio_tensor = torch.randn(1, 16000)
    transcription = asr.transcribe(audio_tensor)
