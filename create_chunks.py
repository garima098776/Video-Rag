import whisper
import json
import os 

model = whisper.load_model("base")

os.makedirs("jsons", exist_ok=True)

audios = os.listdir("audios")

for audio in audios:
    if not (audio.endswith(".mp3") or audio.endswith(".mp4")):
                continue


    if("_" in audio):


        number = audio.split("_")[0]
        title = audio.split("_")[1].replace(".mp3", "")
        print(number, title) 
        result = model.transcribe(f"audios/{audio}",
                          language="hi",
                          task="translate",
                          word_timestamps=False)
         
                        

        chunks = []
        for segment in result["segments"]:
            chunks.append({"number": number, "title": title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})

        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata,f)    
