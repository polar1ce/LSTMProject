import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Model import LSTM  
import hashlib
import hmac
import uuid
import rsa 

# 初始化模型和数据
def initialize_model_and_data():
    INPUT_SIZE = 3
    HIDDEN_SIZE = 32
    NUM_LAYERS = 3
    OUTPUT_SIZE = 1

    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

    train_x = torch.load('Dataset/data1.mat')
    train_y = torch.load('Dataset/label1.mat')

    return lstm, train_x, train_y

# 训练模型
def train_model(lstm, train_x, train_y):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=1e-2)

    max_epochs = 500
    loss_list = []
    epoch_list = []

    for epoch in range(max_epochs):
        output = lstm(train_x)
        loss = loss_function(output, train_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if loss.item() < 1e-5:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
        epoch_list.append(epoch + 1)
        loss_list.append(loss.item())

    return loss_list, epoch_list

# 保存模型
def save_model(lstm):
    torch.save(lstm, 'Model/model1.mat')

# 加载模型
def load_model():
    return torch.load('Model/model1.mat')

# 测试模型
def test_model(lstm, test_x, test_y):
    predicted_y = lstm(test_x)

    loss_function = nn.MSELoss()
    loss = loss_function(predicted_y, test_y)

    threshold = 0.5
    predicted_labels = (predicted_y > threshold).int()
    correct_predictions = (predicted_labels == test_y)
    acc = correct_predictions.sum().float() / test_y.size(0) * 100

    return loss.item(), acc

# 画图
def plot_loss_curve(loss_list):
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    start_time = time.time()

    lstm, train_x, train_y = initialize_model_and_data()
    loss_list, epoch_list = train_model(lstm, train_x, train_y)
    save_model(lstm)

    print("...Training Finished...")

    test_x = torch.load('Dataset/data2.mat')
    test_y = torch.load('Dataset/label2.mat')

    lstm = load_model()
    test_loss, test_acc = test_model(lstm, test_x, test_y)
    end_time = time.time()

    print('Test Loss: {:.5f}'.format(test_loss))
    print('Test Accuracy: {:.2f}%'.format(test_acc))
    print("...Test Finished...")
    execution_time = end_time - start_time
    print(f"函数运行时间: {execution_time}秒")

    plot_loss_curve(loss_list)

