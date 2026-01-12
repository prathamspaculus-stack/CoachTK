import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
import re

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

loader = TextLoader("Z15.YT.time.txt", encoding="utf-8")
docs = loader.load()
full_text = docs[0].page_content


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(full_text)

source_type = "Youtube"
reference_link = "https://www.youtube.com/watch?v=xsZNaePbbTg&t=13s"

prompt = PromptTemplate(
    template="""
You are an expert content analyst.

The input text may contain timestamps in this exact format:
[MM:SS - MM:SS]

TASK:
1. REMOVE all timestamps from the text.
2. EXTRACT the FIRST timestamp EXACTLY as it appears .
3. EXTRACT the LAST timestamp EXACTLY as it appears.

IMPORTANT:
- Do NOT summarize, explain, or rewrite the text.
- Preserve original wording.
- Return ONLY valid JSON.
- Do NOT include markdown or comments.

RULES:
- domain must be ONE of: Leadership, Mindset, IT, Strategy
- topic must be 1–3 short words
- content_type must be ONE of: Framework, Example, Story, Advice
- If no timestamp exists, timestamp must be null

FIELDS:
- domain
- topic
- content_type
- first_timestamp
- last_timestamp
- cleaned_text


TEXT:
{text}

""",
    input_variables=["text"]
)


def safe_json_load(text: str):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group())

def combine_timestamps(first_ts, last_ts):

    if not first_ts or not last_ts:
        return None

    first_ts = first_ts.strip("[]")
    last_ts = last_ts.strip("[]")

    start_time = first_ts.split(" - ")[0]
    end_time = last_ts.split(" - ")[1] 

    return f"{start_time} - {end_time}"

def remove_timestamps(text: str) -> str:
    return re.sub(
        r"\[\d{2}:\d{2}\s*-\s*\d{2}:\d{2}\]",
        "",
        text
    ).strip()



parser = StrOutputParser()
chain = prompt | model | parser


processed_chunks = []

for i, chunk in enumerate(chunks):

    
    raw_metadata = chain.invoke({"text": chunk})
    metadata = safe_json_load(raw_metadata)
    

    if isinstance(metadata, list):
        metadata = metadata[0]

    combined_timestamp = combine_timestamps(
        metadata.get("first_timestamp"),
        metadata.get("last_timestamp")
    )

    metadata["timestamp"] = combined_timestamp  

    metadata.pop("first_timestamp", None)
    metadata.pop("last_timestamp", None)

    metadata["reference_link"] = reference_link
    metadata["source_type"] = source_type

    raw_text = metadata.pop("cleaned_text", chunk)
    cleaned_text = remove_timestamps(raw_text)


    processed_chunks.append({
        "chunk_id": f"chunk_{i+1}",
        "text": cleaned_text,
        "metadata": metadata
    })


with open("Z15.YT.json", "w", encoding="utf-8") as f:
    json.dump(processed_chunks, f, indent=2, ensure_ascii=False)

print("File saved as: processed_chunks.json")



