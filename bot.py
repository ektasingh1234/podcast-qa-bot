import groq
import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import os

# ---- Load Transcript with LARGER chunks ----
def load_transcript(file_path):
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Clean lines first
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line:
            match = re.match(r'\[(\d+)s\] (.*)', line)
            if match:
                clean_lines.append((int(match.group(1)), match.group(2)))

    # Build overlapping chunks of 10 lines with 5 line stride for better coverage
    for i in range(0, len(clean_lines), 5):
        group = clean_lines[i:i+10]
        if not group:
            continue
        start_time = group[0][0]
        text = ' '.join([g[1] for g in group])
        chunks.append({'text': text, 'timestamp': start_time})

    return chunks

print("Loading transcript...")
chunks = load_transcript('transcript.txt')
print(f"Total chunks: {len(chunks)}")

print("Building search index...")
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [c['text'] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print("Ready!")

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def answer_question(question):
    if not question or not question.strip():
        return """<div style="font-family:'Inter',sans-serif;padding:24px;text-align:center;color:#666;">
            <p style="font-size:15px;">👆 Type a question above or click a suggestion, then press <b>Ask 🚀</b></p>
        </div>"""

    q_embedding = model.encode([question])
    # Search top 30 candidates
    distances, indices = index.search(np.array(q_embedding), k=30)

    seen_times = set()
    top_chunks = []
    for idx, dist in zip(indices[0], distances[0]):
        chunk = chunks[idx]
        t = chunk['timestamp']
        # Keep timestamps at least 3 minutes apart
        too_close = any(abs(t - seen_t) < 180 for seen_t in seen_times)
        if not too_close:
            seen_times.add(t)
            top_chunks.append({'text': chunk['text'], 'timestamp': t, 'dist': dist})
        if len(top_chunks) == 3:
            break

    # Already sorted by distance (best first) since FAISS returns in order
    context = '\n\n'.join([c['text'] for c in top_chunks])

    client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""You are a helpful assistant trained on the Elon Musk x Nikhil Kamath podcast (People by WTF Ep. 16).

Context from podcast (these are the most relevant sections):
{context}

Question: {question}

Answer based only on what is said in the podcast context above. Be specific and direct. If the context doesn't clearly answer it, say so honestly."""
        }]
    )

    answer = response.choices[0].message.content

    # Rank scores properly: best = lowest distance
    max_dist = max(c['dist'] for c in top_chunks) or 1
    labels = ["🥇 Best Match", "🥈 2nd Match", "🥉 3rd Match"]
    bg_colors = ["#c0392b", "#1e1e2e", "#1e1e2e"]
    border_colors = ["#e74c3c", "#444", "#444"]
    timestamp_buttons = ""

    for i, chunk in enumerate(top_chunks):
        t = chunk['timestamp']
        # Score: best match (lowest dist) gets highest %
        score = round((1 - chunk['dist'] / (max_dist + 0.01)) * 100)
        score = max(score, 10)
        time_str = format_time(t)
        link = f"https://youtu.be/Rni7Fz7208c?t={t}"
        timestamp_buttons += f"""
        <a href="{link}" target="_blank" style="
            display:inline-block;
            margin:6px 8px 6px 0;
            padding:11px 20px;
            background:{bg_colors[i]};
            color:white;
            border-radius:10px;
            text-decoration:none;
            font-size:14px;
            font-weight:500;
            border:1px solid {border_colors[i]};
            font-family:'Inter',sans-serif;
        ">{labels[i]} — {time_str} ({score}% match)</a>"""

    return f"""
    <div style="font-family:'Inter',sans-serif; padding:24px; line-height:1.8; background:#12121f; border-radius:14px; margin-top:12px; border:1px solid #2a2a3e;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:14px;">
            <span style="font-size:20px;">🎙️</span>
            <h3 style="color:#ff4444; margin:0; font-size:18px; font-weight:700;">Answer</h3>
        </div>
        <p style="font-size:15px; color:#ddd; margin:0 0 20px 0; line-height:1.9;">{answer}</p>
        <hr style="border:none; border-top:1px solid #2a2a3e; margin:16px 0;">
        <p style="color:#777; font-size:13px; margin:0 0 12px 0; font-weight:500;">▶️ Jump to these moments in the podcast:</p>
        <div style="display:flex; flex-wrap:wrap; gap:4px;">{timestamp_buttons}</div>
    </div>"""

def set_question(q):
    return q

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Elon x Nikhil Podcast Bot",
    css="""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body, .gradio-container { font-family: 'Inter', sans-serif !important; }
        .gradio-container { max-width: 880px !important; margin: auto !important; }
        footer { display: none !important; }
        .suggest-btn button {
            font-size: 13px !important;
            width: 100% !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            border-radius: 10px !important;
            padding: 13px 10px !important;
            background: #1a1a2e !important;
            border: 1px solid #2e2e4e !important;
            color: #bbb !important;
        }
        .suggest-btn button:hover {
            background: #22223a !important;
            border-color: #ff4444 !important;
            color: white !important;
        }
        .ask-btn button {
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 15px !important;
            font-family: 'Inter', sans-serif !important;
        }
        input[type="text"], textarea {
            font-family: 'Inter', sans-serif !important;
            font-size: 15px !important;
        }
    """
) as demo:

    gr.HTML("""
        <div style="text-align:center; padding:36px 0 18px;">
            <div style="font-size:52px; margin-bottom:10px;">🎙️</div>
            <h1 style="font-size:32px; font-weight:700; margin:0 0 8px 0; font-family:'Inter',sans-serif; letter-spacing:-0.5px;">
                Elon x Nikhil Podcast Q&A Bot
            </h1>
            <p style="color:#666; font-size:14px; font-family:'Inter',sans-serif; margin:0;">
                Ask anything from the podcast — get an AI answer + clickable YouTube timestamps
            </p>
        </div>
    """)

    gr.HTML("<p style='color:#777; font-size:13px; margin:4px 0 10px 0; font-family:Inter,sans-serif; font-weight:500;'>💡 Click a suggestion to fill the box, then press Ask:</p>")

    with gr.Row():
        b1 = gr.Button("🔍  What is first principles thinking?", elem_classes="suggest-btn")
        b2 = gr.Button("🚀  How did Elon start SpaceX?", elem_classes="suggest-btn")
        b3 = gr.Button("💪  What does Elon think about failure?", elem_classes="suggest-btn")
    with gr.Row():
        b4 = gr.Button("🏢  How does Elon manage multiple companies?", elem_classes="suggest-btn")
        b5 = gr.Button("💡  What is Elon's advice for entrepreneurs?", elem_classes="suggest-btn")

    gr.HTML("<hr style='border:none; border-top:1px solid #1e1e2e; margin:20px 0 16px 0;'>")

    with gr.Row(equal_height=True):
        question_box = gr.Textbox(
            placeholder="Type your question here or click a suggestion above...",
            label="Your Question",
            scale=5,
            lines=1
        )
        ask_btn = gr.Button("Ask 🚀", variant="primary", scale=1, min_width=120, elem_classes="ask-btn")

    output = gr.HTML()

    gr.HTML("""
        <div style="text-align:center; padding:24px 0 8px; color:#333; font-size:12px; font-family:'Inter',sans-serif;">
            Built with yt-dlp &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; SentenceTransformers &nbsp;·&nbsp; Groq LLaMA 3.3 &nbsp;·&nbsp; Gradio
        </div>
    """)

    b1.click(fn=set_question, inputs=gr.State("What is first principles thinking?"), outputs=question_box)
    b2.click(fn=set_question, inputs=gr.State("How did Elon start SpaceX?"), outputs=question_box)
    b3.click(fn=set_question, inputs=gr.State("What does Elon think about failure?"), outputs=question_box)
    b4.click(fn=set_question, inputs=gr.State("How does Elon manage multiple companies?"), outputs=question_box)
    b5.click(fn=set_question, inputs=gr.State("What is Elon's advice for entrepreneurs?"), outputs=question_box)

    ask_btn.click(fn=answer_question, inputs=question_box, outputs=output)
    question_box.submit(fn=answer_question, inputs=question_box, outputs=output)

demo.launch()
