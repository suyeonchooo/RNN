import sys
import os

# 현재 파일의 경로 기준으로 상위 디렉터리(RNN 폴더 경로) 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import data         # RNN 폴더 안의 data.py 모듈을 가져오기
import model

EPOCHS = 1
for epoch in range(EPOCHS):
    # for batch in range(data.batch_size):
    for batch in range(1):
        # print(len(data.sst_embeddings))
        # print(len(data.sst_embeddings[batch]))
        myRNN = model.RNNModel(data.sst_embeddings[batch])

        prediction = myRNN.forward()
        print("The forward pass was successful.")

        # loss = myRNN.binary_cross_entropy_loss(prediction, data.sst_labels)
        # print("The loss was successfully obtained.")
        # print(f'loss: {loss}')

        # pred = myRNN.prediction(h)
