from transformers import pipeline
import torch
import gradio as gr
import numpy as np 
from typing import Dict, Tuple

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
        bool: determine if there's speech
    """
    return len(get_speech_ts(wav, vad_model, sampling_rate = sr)) > 0


default_state = {"audio": None, "text": "", "in_speech": False}


def transcribe(
        audio: Tuple, 
        state: Dict = default_state.copy()):
        """Transcribe

        Args:
            audio (Tuple): audio from gr.audio
            state (_type_, optional): State. Defaults to {"text": "", "audio": None, "in_speech": False}.

        Returns:
            Tuple: Interface 
        """
        
        
        if state is None:
            state = default_state.copy()
        
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
                trans_text += + "\n"
                state["text"] +=  trans_text + "\n"  
                
                state["audio"] = None
                state["in_speech"] = None
                
                
        waveform  = gr.make_waveform(audio)
        return state["text"], waveform, int(speech), state

theme = gr.themes.Soft()

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        """
        # Microphone stream
        Try the microphone and see your words transcribed!
        """        
        )
    
    micro_input = gr.Audio(
        sources= ["microphone"],
        streaming=True,
        label = "Activate Microphone")
    
    gr.Markdown(
        """
        ## Output
        Transcriptions, last transcripted audio and VAT
        """
    )
    with gr.Row():
        text = gr.Textbox(
            label= "Transcription")
        
        video = gr.Video(label= "Audio")
        vad = gr.Radio(choices= [("True", 1), ("False", 0)], label= "VAD")
    
    
    inputs = [micro_input, gr.State()]
    outputs = [text, video, vad, gr.State()]
    
    micro_input.change(transcribe, inputs, outputs, show_progress= "hidden")

# demo = gr.Interface(
#     fn=transcribe, 
#     inputs=[
#         gr.Audio(sources=["microphone"], streaming=True, show_label=False),
#         "state",
#     ],
#     outputs=[
#         "textbox",
#         "state",
#         "video",
#         gr.Radio(choices= [("True", 1), ("False", 0)], label= "Speech"),
#         gr.Radio(choices= [("True", 1), ("False", 0)], label= "in speech")
#     ],
#     theme = theme,
#     live=True)

demo.launch(server_name="0.0.0.0")