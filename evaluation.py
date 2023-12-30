import time
import numpy as np
import torch

import feature_extraction
import myconfig
import neural_net

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


if __name__ == "__main__":
   
   file1, file2 = "./data/sunny1.wav","./data/sunny.wav"

   scores = compare_two(file1, file2)
   print(scores)