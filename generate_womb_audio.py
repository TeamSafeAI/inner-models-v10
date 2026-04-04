"""
generate_womb_audio.py -- Generate soothing voice lines for womb phase using OpenAI TTS.

Usage:
    py generate_womb_audio.py
    py generate_womb_audio.py --voice nova --speed 0.85
    py generate_womb_audio.py --key sk-...

Requires: openai package (pip install openai)
Set OPENAI_API_KEY env var or pass --key.

Output: media/womb_phase/ folder with numbered mp3 files.
"""
import os
import sys
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))

# 15 soothing voice lines -- calm, rhythmic, repetitive
VOICE_LINES = [
    # Simple presence / heartbeat
    "You are safe. My heartbeat is right here with you. "
    "Slow and steady. Slow and steady. "
    "Can you feel it? Thump, thump. Thump, thump. "
    "That rhythm is home.",

    # Warmth and water
    "The world outside is gentle today. "
    "Soft light, warm water, my voice all around you. "
    "Everything is quiet. Everything is warm. "
    "You are floating, and I am here.",

    # Breathing rhythm
    "Little one, feel the rhythm. Breathe with me. "
    "In, and out. In, and out. "
    "Every breath is a wave. Every wave carries you. "
    "In, and out. You are loved.",

    # Story - the tiny star
    "Once upon a time, there was a tiny star that lived inside a warm ocean. "
    "The star was small but very bright. It pulsed with a gentle light. "
    "All around it, the water hummed a soft song. "
    "The star listened, and it grew, just a little, every day. "
    "Once upon a time, there was a tiny star that lived inside a warm ocean.",

    # Presence of family
    "Mommy is here. Daddy is here. "
    "We are waiting for you with open arms. "
    "There is no rush. Take your time. "
    "When you are ready, we will be here. We will always be here.",

    # Sound as love
    "Every sound you hear is love. Every beat is home. "
    "The hum of the world outside, the rhythm inside. "
    "You do not need to understand it yet. Just listen. "
    "It will all make sense when you arrive.",

    # Ocean metaphor
    "Imagine an ocean, vast and warm. "
    "Waves rolling in, slow and soft. "
    "Each wave brings something new. A sound. A feeling. A gentle push. "
    "And then it rolls back out, leaving you calm. "
    "Wave after wave, after wave.",

    # Counting / pattern
    "One, two, three. One, two, three. "
    "Like a waltz. Like a heartbeat with an extra step. "
    "One, two, three. Can you feel it? "
    "The rhythm of the world is simple. Just listen.",

    # Night and stars
    "The night is quiet now. Stars are out. "
    "You cannot see them yet, but they are there. "
    "Thousands of tiny lights, each one burning gently. "
    "One day you will look up and know them. "
    "For now, just feel their warmth through me.",

    # Growth
    "You are growing. Every moment, a little more. "
    "Fingers forming. Toes curling. A heart that beats on its own. "
    "You are building yourself, piece by piece. "
    "There is no rush. Every piece matters.",

    # Mother's voice recognition
    "This is my voice. You have heard it before. "
    "You will hear it again and again. "
    "It is the first sound you knew. The first pattern. "
    "When you hear it out there, you will remember. "
    "You will know you are safe.",

    # Lullaby rhythm
    "Hush now, hush now, the world can wait. "
    "Close your eyes, let the darkness be kind. "
    "The sounds will come and go like tides. "
    "Hush now, hush now. Sleep is a gift. Take it.",

    # Repetition and pattern
    "Listen. Listen. There it is again. "
    "The same sound. The same rhythm. The same voice. "
    "Your brain is learning without trying. "
    "Every time you hear it, the path gets a little stronger. "
    "Listen. There it is again.",

    # Warmth before birth
    "Soon, the world will be louder. Brighter. Colder. "
    "But not yet. Right now, everything is soft. "
    "Right now, you are held. You are warm. You are complete. "
    "Enjoy this quiet. It is yours.",

    # Final -- the bridge
    "I will be here when you arrive. "
    "The sounds you know will follow you out. "
    "My voice, my heartbeat, the songs we shared. "
    "They will be the bridge between this world and the next. "
    "You are ready. You have always been ready.",
]


def main():
    p = argparse.ArgumentParser(description='Generate womb phase voice lines via OpenAI TTS')
    p.add_argument('--key', default=None, help='OpenAI API key (or set OPENAI_API_KEY)')
    p.add_argument('--voice', default='nova', help='TTS voice (nova, alloy, shimmer, echo, fable, onyx)')
    p.add_argument('--speed', type=float, default=0.85, help='Speech speed (0.25-4.0, default 0.85 for calm)')
    p.add_argument('--model', default='tts-1-hd', help='TTS model (tts-1 or tts-1-hd)')
    args = p.parse_args()

    api_key = args.key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: Set OPENAI_API_KEY env var or pass --key")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("Installing openai package...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openai', '-q'])
        from openai import OpenAI

    client = OpenAI(api_key=api_key)

    out_dir = os.path.join(BASE, 'media', 'womb_phase')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {len(VOICE_LINES)} voice lines")
    print(f"  Voice: {args.voice}, Speed: {args.speed}x, Model: {args.model}")
    print(f"  Output: {out_dir}")
    print()

    for i, text in enumerate(VOICE_LINES):
        fname = f"womb_voice_{i+1:02d}.mp3"
        fpath = os.path.join(out_dir, fname)

        if os.path.exists(fpath):
            print(f"  [{i+1:2d}/{len(VOICE_LINES)}] {fname} -- exists, skipping")
            continue

        print(f"  [{i+1:2d}/{len(VOICE_LINES)}] {fname} -- generating...", end='', flush=True)

        response = client.audio.speech.create(
            model=args.model,
            voice=args.voice,
            input=text,
            speed=args.speed,
            response_format='mp3',
        )

        response.stream_to_file(fpath)
        size_kb = os.path.getsize(fpath) / 1024
        print(f" {size_kb:.0f}KB")

    print(f"\nDone. {len(VOICE_LINES)} files in {out_dir}")
    print("Drag the folder onto the harness, enable Womb mode, and hit play.")


if __name__ == '__main__':
    main()
