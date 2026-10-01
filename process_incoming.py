import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib
import requests
# from openai import OpenAI
# from config import api_key

# client = OpenAI(api_key=api_key)
def format_time(seconds):
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes}:{sec:02d}"








def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
   

    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json= {
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
          
    })
    
    response = r.json()
    print(response)
    return(response)


# def inference_openai(prompt):
#     try:
#         response = client.responses.create(
#             model="gpt-5",
#             input=prompt
#         )
#         return response.output_text
#     except Exception as e:
#         return f"OpenAI Error: {e}"





df = joblib.load('embeddings.joblib')



incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]








# similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()

emb_matrix = np.array(df['embedding'].tolist(), dtype=np.float32)

similarities = cosine_similarity(
    emb_matrix,
    [question_embedding]
).flatten()

top_results = 5
max_index = similarities.argsort()[::-1][0:top_results]


# new_df = df.loc[max_index]
new_df = df.iloc[max_index].copy()
new_df = new_df.sort_values("start")


new_df["start"] = new_df["start"].apply(format_time)
new_df["end"] = new_df["end"].apply(format_time)



# print(new_df[['text', 'start']])

# best_idx = similarities[max_index].argmax()
# best_row = new_df.iloc[best_idx]



prompt = prompt = prompt = f"""
You are a strict video timestamp extraction system for a Data Science course.

========================
TASK
========================
Find EXACT start and end time where the user question is explained in the video chunks.

========================
STRICT OUTPUT RULES
========================
- Return ONLY video number, start time, end time
- NO explanation
- NO sentence
- NO extra text
- NO markdown
- NO title
- NO text field
- ONLY JSON array

========================
OUTPUT FORMAT
========================
[
  {{
    "number": "video file name",
    "start": start_time,
    "end": end_time
  }}
]

========================
IMPORTANT
========================
- Choose ONLY the best matching chunk
- Do NOT hallucinate or guess time
- Use only provided chunks

========================
CHUNKS
========================
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

========================
USER QUESTION
========================
{incoming_query}
"""












with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)




