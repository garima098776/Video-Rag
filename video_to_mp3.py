import os
import subprocess

files = os.listdir("vds")
for file in files:
    print(file)
    tutorial_number = file.split(" [")[0].split(" #")[1]
    file_name = file.split(" - ")[0]
    print(tutorial_number, file_name)
    subprocess.run(["ffmpeg","-i",f"vds/{file}",f"audios/{tutorial_number}_{file_name}.mp3"])

    