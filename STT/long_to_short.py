import os 
from pydub import AudioSegment 
from pydub.silence import split_on_silence 
  

class long_to_short():

    def __init__(self):
        pass

    def split(self, input_path, output_path, min_silence=600, silence_threshold = -16):
        # make output directory
        try:
            os.mkdir(output_path)
        except(FileExistsError):
            pass

        # find wav list and split each wav file
        file_lst = [x for x in os.listdir(input_path) if x.endswith('.wav')]
        for f in file_lst:
            self.long_to_short(os.path.join(input_path, f), output_path, min_silence, silence_threshold)


    def long_to_short(self, file, output_path, min_silence, silnece_threshold):
        # load sound file
        sound = AudioSegment.from_file(file)
        dBFS = sound.dBFS
        # split on silence
        chunks = split_on_silence(sound, 
            min_silence_len = min_silence,
            silence_thresh = dBFS + silnece_threshold,
            keep_silence = min_silence*0.8
        )

        # merge chunk files if chunk length is shorter than 2 seconds
        target_length = 2 * 1000
        output_chunks = [chunks[0]]
        for chunk in chunks[1:]:
            if len(output_chunks[-1]) < target_length:
                output_chunks[-1] += chunk
            else:
                # if the last output chunk is longer than the target length,
                # we can start a new one
                output_chunks.append(chunk)

        # save each chunk file
        filename = file.split('/')[-1][:-4]
        for i, chunk in enumerate(output_chunks):
            chunk_name = filename+"_chunk_{0}.wav".format(i)
            print("saving {}".format(chunk_name)) 
            # specify the bitrate to be 192 k 
            chunk.export(output_path + '/' + chunk_name, bitrate ='192k', format ="wav") 

  
  
if __name__ == '__main__':
    lts = long_to_short()
    input_path = '/home/ubuntu/workspace/STT_project_github/STT_project/STT/data/long_to_short'
    output_path = '/home/ubuntu/workspace/STT_project_github/STT_project/STT/data/long_to_short/test'
    lts.split(input_path, output_path)
    
