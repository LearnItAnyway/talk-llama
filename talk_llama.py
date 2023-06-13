import gradio as gr
import torch
import soundfile as sf
from transformers import GPT2LMHeadModel, GPT2Tokenizer, WhisperProcessor, WhisperForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np
from scipy.signal import resample

device='cuda'
# Initialize the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# Load Whisper model and processor
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)
whisper_model.config.forced_decoder_ids = None

# Load SpeechT5 model, processor, and vocoder
speech_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
speech_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# Load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

hist = ""
def text_formatter(role, text):
    global hist
    if role == "human":
        hist += f"### USER: {text}"
    else:
        hist += f"### ASSISTANT: {text}"

def resample_audio(audio, orig_rate=48000, new_rate=16000):
    duration = audio.shape[0] / orig_rate
    new_length = int(duration * new_rate)
    resampled_audio = resample(audio, new_length)
    return resampled_audio

# Define function for generating response
def generate_response(input_audio, input_text):
    global hist
    if input_audio is not None:
        input_text = voice_to_text(input_audio)
   #else:
   #    input_audio_path = text_to_voice(input_text)
   #    input_text = voice_to_text(input_audio_path)
    text_formatter('human', input_text)
    inputs = tokenizer.encode(f"{hist}\n### ASSISTANT: ", return_tensors='pt').to(device)[-512*3:]
    outputs = model.generate(inputs, max_length=128, num_return_sequences=1,
                             no_repeat_ngram_size=2, early_stopping=True)
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    text_formatter('assistant', response)
    return hist, text_to_voice(response)

# Define function for voice-to-text conversion
def voice_to_text(file_path, whisper_sampling_rate=16000):
    #ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    #sample = ds[0]["audio"]
    data, samplerate = sf.read(file_path)
    data_ = resample_audio(data, samplerate, whisper_sampling_rate)
    input_features = whisper_processor(data_, sampling_rate=whisper_sampling_rate,
                                       return_tensors="pt").input_features.to(device)
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# Define function for text-to-speech conversion
def text_to_voice(text):
    inputs = speech_processor(text=text, return_tensors="pt")
    speech = speech_model.generate_speech(inputs["input_ids"].to(device), speaker_embeddings, vocoder=vocoder)
    sf.write("speech.wav", speech.detach().cpu().numpy(), samplerate=16000)
    return 'speech.wav'


# Define the Gradio interface
iface = gr.Interface(
            generate_response,
            [gr.inputs.Audio(source="microphone", type="filepath"), gr.outputs.Textbox()],
            [gr.inputs.Textbox(label="Input Text"), gr.outputs.Audio(type="filepath", label="Output Speech")],
            title="Voice Assistant",
            description="A voice assistant that can understand your voice input and generate voice output.",
            interpretation="default"
)

iface.launch()
