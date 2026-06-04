# 🎙️ Elon x Nikhil Podcast Q&A Bot

A Q&A bot trained on the **Elon Musk x Nikhil Kamath podcast (People by WTF Ep. 16)**.  
Ask any question → get an AI-generated answer + clickable YouTube timestamps.

## 🚀 What it does
- Extracts the full podcast transcript with timestamps using `yt-dlp`
- Builds a semantic search index using FAISS + SentenceTransformers
- Answers questions using Groq's LLaMA 3.3 (free)
- Returns top 3 clickable YouTube timestamp links for each answer
- Clean Gradio UI with suggested questions

## 🛠️ Tech Stack
| Component | Tool |
|---|---|
| Transcript extraction | yt-dlp |
| Transcript cleaning | Python (regex) |
| Semantic search | FAISS + sentence-transformers |
| LLM | Groq LLaMA 3.3 70B (free) |
| UI | Gradio |

## ⚙️ How to run

1. Clone the repo
2. Install dependencies:
   pip install yt-dlp sentence-transformers faiss-cpu gradio groq
3. Add your Groq API key (free at console.groq.com):
   export GROQ_API_KEY=your_key_here
4. Get the transcript:
   python convert.py
5. Run the bot:
   python bot.py
6. Open http://127.0.0.1:7860

## 📁 Files
- `bot.py` — main app
- `convert.py` — VTT to transcript converter
- `transcript.txt` — cleaned transcript with timestamps

## 🔍 How it works
1. YouTube auto-captions downloaded as `.vtt` file
2. Cleaned and converted to `[timestamp] text` format
3. Text split into overlapping chunks of 10 lines
4. Each chunk embedded using `all-MiniLM-L6-v2`
5. User question embedded and compared against all chunks via FAISS
6. Top 3 most relevant chunks passed to LLaMA 3.3 as context
7. Answer generated + YouTube links built with `?t=seconds` format

## ⚠️ Limitations
- Auto-captions are ~85-90% accurate (Whisper would be better)
- Timestamp points to start of relevant chunk, not exact sentence
- Answer quality depends on how well the question matches transcript chunks
- Works only for this specific podcast episode
