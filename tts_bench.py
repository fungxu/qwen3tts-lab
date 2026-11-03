import os,time,torch,soundfile as sf
from qwen_tts import Qwen3TTSModel
dev=os.environ.get("DEV","mps"); dt=os.environ.get("DT","float32")
m={"float32":torch.float32,"bfloat16":torch.bfloat16}[dt]
t0=time.time(); model=Qwen3TTSModel.from_pretrained("models/Qwen3-TTS-12Hz-1.7B-CustomVoice",device_map=dev,dtype=m)
print(f"[load] {dev}/{dt} {time.time()-t0:.2f}s",flush=True)
text="其实我真的有发现，我是一个特别善于观察别人情绪的人。"
for i in range(3):
    t1=time.time()
    with torch.inference_mode():
        wavs,sr=model.generate_custom_voice(text=text,speaker="Vivian",language="Chinese")
    gen=time.time()-t1; dur=len(wavs[0])/sr
    print(f"[run{i}] audio={dur:.2f}s gen={gen:.2f}s rtf={gen/dur:.2f}x",flush=True)
    if i==2: sf.write(f"out-{dev}-{dt}-run2.wav",wavs[0],sr)
