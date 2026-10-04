from gtts import gTTS
import edge_tts
import asyncio

def generate_english_audio(text, output_path="../outputs/sentence_en.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(output_path)

async def _generate_german_audio_async(text, output_path="../outputs/sentence_de.mp3"):
    communicate = edge_tts.Communicate(text, voice="de-DE-KatjaNeural")
    await communicate.save(output_path)

def generate_german_audio_sync(text, output_path="../outputs/sentence_de.mp3"):
    asyncio.run(_generate_german_audio_async(text, output_path))