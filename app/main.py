from transformers import pipeline
import torch
import gradio as gr
import numpy as np 
from typing import Dict


transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base.en")

vad_model, vad_utils = torch.hub.load(
    repo_or_dir= "snakers4/silero-vad",
    model = "silero_vad")

(get_speech_ts, *_) = vad_utils

def is_speech(wav:np.array, sr: int) -> bool:
    """Determines if there's speech on the numpy array

    Args:
        wav (np.array): np.array with data 
        sr (int): sampling rate

    Returns:
        bool: _description_
    """
    return len(get_speech_ts(wav, vad_model, sampling_rate = sr)) > 0

def transcribe(
    audio, 
    state: Dict = {"text": "", "audio": None, "in_speech": False}):
    
    if state is None:
        state = {"text": "", "audio": None, "in_speech": False}
    
    sr, y = audio
    y = y.astype(np.float32)
    
    if np.max(np.abs(y)) != 0:
        y /= np.max(np.abs(y)) 
    
    speech = is_speech(y, sr)
    in_speech = state["in_speech"]
    
    if speech:
        if(state["audio"] is None):
            state["audio"] = y
        else:
            state["audio"] = np.concatenate((state["audio"], y))
        
        if not in_speech:
            state["in_speech"] = True
    
    else:
        if in_speech:
            trans_text = transcriber({"sampling_rate": sr, "raw": state["audio"]})["text"]
            state["text"] += trans_text + "\n"
            state["in_speech"] = False
            
            state["audio"] = None
    waveform  = gr.make_waveform(audio)
    return state["text"], state, waveform, int(speech), int(state["in_speech"])

demo = gr.Interface(
    fn=transcribe, 
    inputs=[
        gr.Audio(sources=["microphone"], streaming=True, show_label=False),
        "state",
    ],
    outputs=[
        "textbox",
        "state",
        "video",
        gr.Radio(choices= [("True", 1), ("False", 0)], label= "Speech"),
        gr.Radio(choices= [("True", 1), ("False", 0)], label= "in speech")
    ],
    live=True)

demo.launch(server_name="0.0.0.0",)