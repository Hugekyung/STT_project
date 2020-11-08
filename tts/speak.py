# pyaudio 설치 필요
import speech_recognition as sr
from gtts import gTTS
import os
import time
import playsound

def speak(text):
    tts = gTTS(text=text, lang='ko')
    filename = 'sample2.mp3' # 음성파일명
    tts.save(filename)
    playsound.playsound(filename)

if __name__ == "__main__":
    text = '텍스트를 입력해주세요'
    speak(text)