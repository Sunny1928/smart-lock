import time
import numpy as np
import torch
import time
import json
import sounddevice as sd
from scipy.io.wavfile import write

import utils.feature_extraction as feature_extraction
import utils.myconfig as myconfig
import utils.neural_net as neural_net
# import utils.audio_file

def run_inference(features, encoder, full_sequence=myconfig.USE_FULL_SEQUENCE_INFERENCE):
    """Get the embedding of an utterance using the encoder."""
    if full_sequence:
        # Full sequence inference.
        batch_input = torch.unsqueeze(torch.from_numpy(features), dim=0).float().to(myconfig.DEVICE)
        batch_output = encoder(batch_input)
        
        return batch_output[0, :].cpu().data.numpy()
    
    else:
        sliding_windows = feature_extraction.extract_sliding_windows(features)
        if not sliding_windows:
            return None
        batch_input = torch.from_numpy(np.stack(sliding_windows)).float().to(myconfig.DEVICE)
        batch_output = encoder(batch_input)

        # Aggregate the inference outputs from sliding windows.
        aggregated_output = torch.mean(batch_output, dim=0, keepdim=False).cpu()
        
        return aggregated_output.data.numpy()


def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings."""

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def eval(user_name):
    """Run evaluation of the saved model on test data."""
    encoder = neural_net.get_speaker_encoder(r'./saved_model/saved_model_lstm.pt')
    waveform = record_audio()
    
    # 讀檔
    all_register_users = load_register_user()
    if user_name not in all_register_users.keys():
        print("Not find this user")
        return 
    target_user_embedding = all_register_users[user_name]
    start_time = time.time()
    
    features = feature_extraction.extract_features_eval(waveform)
    embedding = run_inference(features, encoder)
    scores = cosine_similarity(embedding, target_user_embedding)

    eval_time = time.time() - start_time
    print("Device: ", myconfig.DEVICE)
    print("Finished evaluation in ", eval_time, "seconds")
    print("Finished scores", scores)

    if scores >= 0.9:
        return True

    return False

def compare_two(file1, file2):
    """Run evaluation of the saved model on test data."""
    start_time = time.time()

    encoder = neural_net.get_speaker_encoder(r'./saved_model/saved_model_lstm.pt')

    features1 = feature_extraction.extract_features(file1)
    features2 = feature_extraction.extract_features(file2)
    
    embedding1 = run_inference(features1, encoder)
    embedding2 = run_inference(features2, encoder)
    
    scores = cosine_similarity(embedding1, embedding2)

    eval_time = time.time() - start_time
    print("Device: ", myconfig.DEVICE)
    print("Finished evaluation in ", eval_time, "seconds")
    print("Finished scores", scores)

    if scores >= 0.8:
        return True

    return False

def register_user(user_name):
    # class 後可以拿掉 Line72
    encoder = neural_net.get_speaker_encoder(r'./saved_model/saved_model_lstm.pt')
    waveform = record_audio()
    feature = feature_extraction.extract_features_eval(waveform= waveform)
    embedding = run_inference(feature, encoder)
    save_embedding(user_name= user_name, embedding= embedding)

def load_register_user():
    with open("./register_user.json", "r") as f:
        return json.load(f)

def save_embedding(user_name, embedding):

    
    try:
        with open("./register_user.json", "r") as f:
            data = json.load(f)
        # Add new user
        data[user_name] = embedding
        json.dump(data, open("./register_user.json",'w'))
    
    except Exception as e:
        """ Create new one json file"""
        print(e)
        print(f"No file ./register_user.json")
        print("Create New One")
        json.dump({user_name: embedding.tolist()}, open("./register_user.json",'w'))
    
    print(f"Successfully save user: {user_name} information ")
        

def record_audio(record_seconds = 4):
    # 此設定與feature.extraction.py 的 sf.read 相同
    sample_rate = 44100  
    channels = 1

    print("Start recording...")
    waveform = sd.rec(int(record_seconds * sample_rate), samplerate= sample_rate, channels= channels, dtype='float64')
    sd.wait()  # 等待錄音結束
    threshold = 0.01
    recording_int16 = np.int16(waveform * 32767)

    # # 將錄音數據保存為 WAV 檔案
    write("./data/test.wav" , sample_rate, recording_int16)
    print("Stop recording...")
    print()
    waveform_1D = waveform.ravel()
    
    # remove silence  
    # first_non_zero = np.where(waveform_1D > 0.01)[0][0]
    # last_non_zero = np.where(waveform_1D > 0.01)[0][-1]
    # non_zero_audio = waveform_1D[first_non_zero:last_non_zero+1]
 
    # # 將錄音數據保存為 WAV 檔案
    # new_list = np.array(non_zero_audio).reshape(-1,1)
    # recording_int16 = np.int16(new_list * 32767)
    # write("./data/test_1.wav" , sample_rate, recording_int16)
    
    return waveform_1D




if __name__ == "__main__":
    file1, file2 = "./data/sunny.wav","./data/test.wav"
    scores = compare_two(file1, file2)
    print(scores)