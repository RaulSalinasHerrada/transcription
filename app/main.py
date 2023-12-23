from transformers import pipeline
import torch
import gradio as gr
import numpy as np 

transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base.en")


vad_model, vad_utils = torch.hub.load(
    repo_or_dir= "snakers4/silero-vad",
    model = "silero_vad")

(
    get_speech_ts,
    _,read_audio, *_) = vad_utils

def is_speech(wav:np.array, sr: int):
    return len(get_speech_ts(wav, vad_model, sampling_rate = sr)) > 0

def transcribe(
    audio, 
    state = {"text": "", "audio": None, "in_speech": False}):
    
    if state is None:
        state = {"text": "", "audio": None, "in_speech": False}
    
    sr, y = audio
    y = y.astype(np.float32)
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
            state["in_speech"] = False
            state["text"] += trans_text + "\n"
            state["audio"] = None
    
    # if speech:
        
    #     if(state["audio"] is None):
    #         state["audio"] = y
    #     else:
    #         state["audio"] = np.concatenate((state["audio"], y))
        
    #     trans_text = transcriber({"sampling_rate": sr, "raw": state["audio"]})["text"]
    #     text =  trans_text + "\n"
    #     state["temp_text"] = text
    
    # else:
    #     state["text"] += state["temp_text"]
    #     state["temp_text"] = ""
    #     state["audio"] = None
    
    return state["text"], state, int(speech)

demo = gr.Interface(
    fn=transcribe, 
    inputs=[
        gr.Audio(sources=["microphone"], streaming=True),
        "state",
    ],
    outputs=[
        "textbox",
        "state",
        gr.Radio(choices= [("Speech", 1), ("No speech", 0)])
    ],
    live=True)

demo.launch(server_name="0.0.0.0")