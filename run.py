import utils.evaluation as evaluation

if __name__ == '__main__':

    file1, file2 = "./data/sunny.wav","./data/test.wav"
    scores = evaluation.compare_two(file1, file2)
    print(scores)
    