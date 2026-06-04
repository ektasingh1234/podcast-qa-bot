import re

def vtt_to_transcript(vtt_file, output_file):
    with open(vtt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    transcript = []
    seen_texts = set()
    current_time = None

    for line in lines:
        line = line.strip()

        # Check if it's a timestamp line
        if '-->' in line:
            time_part = line.split('-->')[0].strip()
            h, m, s = time_part.split(':')
            current_time = int(h)*3600 + int(m)*60 + int(float(s))

        # Check if it's a text line (not empty, not timestamp, not WEBVTT header)
        elif current_time is not None and line and not line.startswith('WEBVTT') and not line.startswith('Kind:') and not line.startswith('Language:'):
            # Remove all tags like <c>, <00:00:51.120>, align:start etc
            clean = re.sub(r'<[^>]+>', '', line)
            clean = clean.strip()

            if clean and clean not in seen_texts:
                seen_texts.add(clean)
                transcript.append(f"[{current_time}s] {clean}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(transcript))

    print(f"Done! {len(transcript)} lines written to transcript.txt")

vtt_to_transcript('podcast.en.vtt', 'transcript.txt')