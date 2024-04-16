from Model import LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# Start time
start_time = time.time()



# Initialization
INPUT_SIZE = 3
HIDDEN_SIZE = 10
NUM_LAYERS = 1
OUTPUT_SIZE = 1

lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

train_x = torch.load('Dataset/data1.mat')
train_y = torch.load('Dataset/label1.mat')

predicted_y = lstm(train_x)
threshold = 0.5
predicted_labels = (predicted_y > threshold).int()
correct_predictions = (predicted_labels == train_y)
acc = correct_predictions.sum().float() / train_y.size(0) * 100
print('Initial Accuracy: {:.2f}%'.format(acc.item()))

# Training parameters
loss_function = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters(), lr=1e-2)

# Training

max_epochs = 10000
loss_list=[]
epoch_list=[]

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

lstm = lstm.eval()
torch.save(lstm, 'Model/model1.mat')
print("...Training Finished...")

# Testing
test_x = torch.load('Dataset/data2.mat')
test_y = torch.load('Dataset/label2.mat')

lstm = torch.load('Model/model1.mat')
predicted_y = lstm(test_x)

loss_function = nn.MSELoss()
loss = loss_function(predicted_y, test_y)


threshold = 0.5
predicted_labels = (predicted_y > threshold).int()
correct_predictions = (predicted_labels == test_y)
acc = correct_predictions.sum().float() / test_y.size(0) * 100

# End time
end_time = time.time()
print('Test Loss: {:.5f}'.format(loss.item()))
print('Test Accuracy: {:.2f}%'.format(acc.item()))
print("...Test Finished...")
execution_time = end_time - start_time  # 计算运行时间
print(f"函数运行时间: {execution_time}秒")

# plot
plt.plot(loss_list, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()