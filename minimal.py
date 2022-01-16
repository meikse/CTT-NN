from pathlib import Path
from csv import DictReader
import math
import matplotlib.pyplot as plt

import torch
from torch import nn, tensor, randn, zeros, no_grad
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

## Load data
PATH = Path(__file__).parent.resolve()
FILE = PATH / "data" / "dataBenchmark.csv"

with open(FILE) as file:
    csv_data = DictReader(file, delimiter=',')
    columns={key: [] for key in csv_data.fieldnames[:-2]} # prepare dict with keys
    for row in csv_data:
        for fieldname in csv_data.fieldnames[:-2]:
            value = float(row.get(fieldname))               # get row value for key
            columns.setdefault(fieldname, []).append(value) # store it in the dict

## pytorch domain
x_train, x_valid, y_train, y_valid = map(tensor, (columns['uEst'], columns['uVal'], columns['yEst'], columns['yVal'])) # map list to tensors 

# demonstrate inputs and output in respect to train and valid data
fig_in, ax_in = plt.subplots()
ax_in.plot(x_train)
ax_in.plot(x_valid)

fig_out, ax_out= plt.subplots()
ax_out.plot(y_train)
ax_out.plot(y_valid)

## model
class Logistics(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(1, 10),
                nn.Sigmoid(),
                nn.Linear(10,100),
                nn.Sigmoid(),
#                 nn.Linear(100,100),
#                 nn.Sigmoid(),
                nn.Linear(100,1),
                )
        
    def forward(self, xb):
        return self.model(xb)

def RMS(pred, yb):
    return torch.sqrt(torch.abs(yb - pred)**2)

loss_func = RMS 
model = Logistics()

## wrapping
train_ds = TensorDataset((x_train),(y_train)) # training dataset
train_dl = DataLoader(train_ds)
valid_ds = TensorDataset((x_valid),(y_valid)) # validation dataset
valid_dl = DataLoader(valid_ds)

## parameters
# lr = 1e-2
# epochs = 20

lr = 1e-3
epochs = 10

## training
train = True
if train:
    for epoch in range(epochs):
        for xb,yb in train_dl:
            pred = model(xb) # forwarding
            loss = loss_func(pred, yb)
        
            loss.backward()  # backwarding
            with no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()
        print('epoch:', epoch, '; loss:', loss.item())
#     torch.save(model.state_dict(), PATH)  # permission denied 
else:
    pass
#     model.load_state_dict(torch.load(PATH))
#     model.eval()

## Evaluation
y_valid_est = [model(xb).item() for xb, yb in valid_dl]

# fig, ax = plt.subplots() # demonstrate model accuracy
# ax.plot(y_valid_est)
# ax.plot(y_valid.numpy())

f11 = plt.figure()
plt.plot(y_valid_est, '-b', label='Simulated')
plt.plot(y_valid.numpy(), 'k', label='Measured')
plt.title('Validation Data')
leg = f11.legend();

# it makes no sense to analyze a nn for system identification, just using it as a simple input output black box, we have to make sure not to analyze the difference between the input and ouput, we also have to consider that change of the state. Thus the NN shall also predict the x_dot and x (x2=y). important to understand!
# just the difference between to parameters ist not enough for the NN to understand the behaviour, you need to give the NN also information about the changes (derivatives of the states)

# maybe try your own generated data with some overleight sinsoidal function overlayed to produce an nonlinear system, which you can than estimate with an ANN. pro: yo can provide as much data as needed and the created data does not contain any noise or fluctuations which cause huge indifferences whcih cannot be extrapolated by the simple ANN

# conclusion: weird things can happen which cannot be explained physically anymore. other methods take place to evaluate the results from the builded nn 
