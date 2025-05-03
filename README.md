# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset


## Design Steps

Step 1:
Import necessary libraries.

Step 2:
Load and preprocess the data.

Step 3:
Create input-output sequences.

Step 4:
Convert data to PyTorch tensors.

Step 5:
Define the RNN model.

Step 6:
Train the model using the training data.

Step 7:
Evaluate the model and plot predictions.



## Program
#### Name: K KESAVA SAI
#### Register Number: 212223230105

```Python 
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out





model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


# Train the Model
epochs=40
model.train()
train_losses=[]

for epoch in range(epochs):
    epoch_loss=0
    for x_batch, y_batch in train_loader:
        x_batch,y_batch=x_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        outputs=model(x_batch)
        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    train_losses.append(epoch_loss/len(train_loader))
    print(f"Epoch [{epoch+1}/{epochs}], loss: {train_losses[-1]:.4f}")


```

## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/user-attachments/assets/ce519f9e-8567-4cbc-a316-633ba6879589)


### Predictions 

![image](https://github.com/user-attachments/assets/fcf2bb2a-ab62-4155-905f-20676020e3f8)


## Result


