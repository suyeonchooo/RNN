import torch
import numpy as np
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = ["0", "1"]
predictions = []

class RNNModel():
    def __init__(self, input):
        self.input = input.to(device)

        # Size
        # input 구성(batch_size, sequence_length, feature_size) 
        self.batch_size = self.input.size()[0]
        self.sequence_length = self.input.size()[1]
        self.input_size = self.input.size()[2]

        self.hidden_size = 512
        self.output_size = 2                              # o(negetive) or 1(positive)

        # Weight Initialization
        self.W_hh = torch.randn(self.hidden_size, self.hidden_size, device=device)
        self.W_xh = torch.randn(self.input_size, self.hidden_size, device=device)
        self.W_hy = torch.randn(self.hidden_size, self.output_size, device=device)

        print(f'input: {self.input.size()}')
        print(f'input size: {self.input_size}')
        print(f'batch size: {self.batch_size}')
        print(f'sequence length: {self.sequence_length}')
        print(f'hidden size: {self.hidden_size}')
        print(f'output size: {self.output_size}')

        # print(f'W_hh: {self.W_hh}')
        print(f'W_hh size: {self.W_hh.size()}')
        # print(f'W_xh: {self.W_xh}')
        print(f'W_xh size: {self.W_xh.size()}')
        # print(f'W_hy: {self.W_hy}')
        print(f'W_hy size: {self.W_hy.size()}')
    
    def forward(self):
        hidden_state = torch.zeros(self.batch_size, self.hidden_size).to(device)        # 초기 hidden state

        for t in range(self.sequence_length):
            x_t = self.input[:, t, :]                                                   # batch 전부 가져오고, 각 t에 맞는 sequence 가져오고, feature 전부 가져옴

            h = torch.matmul(hidden_state, self.W_hh)
            x = torch.matmul(x_t, self.W_xh)
            updated_h = torch.tanh(h + x)

            hidden_state = updated_h                                                    # hidden state update 
        print(f'x_t size: {x_t.size()}') 
        print(f'x size: {x.size()}')
        print(f'h size: {h.size()}')         

        logits = torch.matmul(updated_h, self.W_hy)
        print(logits)
        print(logits.size())
        return logits
    
    def binary_cross_entropy_loss(self, pred, labels):
        loss = torch.tensor(0.0, device=device)
        # labels 구성(batch_size, label_size)
        for batch in range(self.batch_size):
            label = torch.tensor(float(labels[batch]), device=device)
            label_index = int(labels[batch])
            print(label)
            print(label_index)
            print(pred[batch][label_index])
            loss += label * torch.log(pred[batch][label_index]) + (1 - label) * torch.log(1- pred[batch][label_index])
            print(loss)
        average_loss = -(loss / self.batch_size)

        return average_loss
       
    def backward (self, learning_rate, loss):
        pass

    def optimizer(self):
        pass
    
    def output_layer(self, logits):
        pass


    # def prediction (self, updated_h):
    #     logits = torch.matmul(updated_h, self.W_hy)

    #     softmax = nn.Softmax(dim=1)
    #     y = softmax(logits)

    #     pred_index = torch.argmax(y)
    #     pred = labels[pred_index.item()]               # pred_index를 정수로 변환
    #                                                    # torch.argmax()는 tensor로 결과를 반환 > .item()이 tensor에서 scalar 값을 추출해줌
    #                                                    # (tensor(1)에서 1이라는 값을 빼내는 역할을 하는 것)
    #     predictions.append(pred)
    #     return predictions