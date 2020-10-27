# https://www.geeksforgeeks.org/python-speech-recognition-on-large-audio-files/

# importing libraries 
import os 
  
from pydub import AudioSegment 
from pydub.silence import split_on_silence 
  
# a function that splits the audio file into chunks 
# and applies speech recognition 
def silence_based_conversion(path='./test.wav'): 
  
    # open the audio file stored in 
    # the local system as a wav file. 
    wav = AudioSegment.from_wav(path) 
  
    # split track where silence is 0.5 seconds  
    # or more and get chunks 
    chunks = split_on_silence(wav, min_silence_len = 500, silence_thresh = -16) 

    # create a directory to store the audio chunks. 
    try: 
        os.mkdir('audio_chunks') 
    except(FileExistsError): 
        pass
  
    # move into the directory to 
    # store the audio files. 
    os.chdir('audio_chunks') 
  
    # process each chunk 
    for i, chunk in enumerate(chunks): 
              
        # Create 0.5 seconds silence chunk 
        chunk_silent = AudioSegment.silent(duration=500) 
  
        # add 0.5 sec silence to beginning and  
        # end of audio chunk. This is done so that 
        # it doesn't seem abruptly sliced. 
        audio_chunk = chunk_silent + chunk + chunk_silent 
  
        # export audio chunk and save it in  
        # the current directory. 
        print("saving chunk{0}.wav".format(i)) 
        # specify the bitrate to be 192 k 
        audio_chunk.export("./chunk{0}.wav".format(i), bitrate ='192k', format ="wav") 
  
        # the name of the newly created chunk 
        filename = 'chunk'+str(i)+'.wav'
  
        print("Processing chunk "+str(i)) 
  
    os.chdir('..') 
  
  
if __name__ == '__main__': 
    print('Enter the audio file path') 
    silence_based_conversion() 