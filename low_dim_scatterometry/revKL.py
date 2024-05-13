import torch
import numpy as np
from torch import nn
from FrEIA.framework import InputNode, OutputNode, Node, GraphINN, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, InvertibleSigmoid
import FrEIA.modules as Fm
from tqdm import tqdm
torch.set_default_dtype(torch.float32)
import matplotlib.pyplot as plt
from utils_high import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DIMENSION = 3
ndim_y = 23

# load forward model
forward_model = nn.Sequential(nn.Linear(DIMENSION, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256,  ndim_y)).to(device)

forward_model.load_state_dict(torch.load('forward_model_new.pt'))
for param in forward_model.parameters():
    param.requires_grad=False

    
    
# define hyperparameters
relu = torch.nn.ReLU()
n_iter_max = 5000
x_total_y = 2000
# number of measurements and number of runs for evaluation
no_meas = [1,2,4,8]
no_runs = 10
dist_list = np.zeros((4))
elbo_list = np.zeros((4))
a_true = 0.005
b_true = 0.1

for p in range(no_runs):

    print(p)
    mean = 0
    elbo_mean = 0
    torch.manual_seed(p)
    x_true = torch.rand(8, DIMENSION, device = device)*2-1
    meas2 = forward_model(x_true)+b_true*torch.randn_like(forward_model(x_true), device = device)*forward_model(x_true)+a_true*torch.randn_like(forward_model(x_true), device = device)
    for q in range(len(no_meas)):
        meas = meas2[:no_meas[q]:,]
        print(meas.shape)

        model = create_INN(4,256)
        a = torch.ones(1,1, device = device).clone().detach()*0.3
        b = torch.ones(1,1, device = device).clone().detach()*0.3

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        max_elbo = 0.
        opt_a = 0.
        opt_b = 0.
        tq = tqdm(range(n_iter_max), desc='elbo')
        for it in tq:
            l,a,b = train_inn_epoch_rev(model, meas, optimizer, a.detach(), b.detach(),forward_model)
            with torch.no_grad():
                elbo = calc_elbo(a,b,model,meas)
                if elbo > max_elbo:
                    max_elbo = elbo
                    torch.save(model.state_dict(), 'rev_KL_model.pt')
                    opt_a = a.clone()
                    opt_b = b.clone()
            tq.set_description(f"elbo {elbo.item()}")
                    
        model.load_state_dict(torch.load('rev_KL_model.pt'))   
        a = opt_a.clone()
        b = opt_b.clone()
        y_meas = torch.zeros(2000,ndim_y,device = device)
        y_meas = y_meas + meas[0]
        z = torch.randn(2000,DIMENSION, device = device)
        samples = model(z,c = y_meas)
        savePost(samples, DIMENSION, x_true[0], p*len(no_meas)+q, 'posterior_rev')
        ab_error = torch.abs((b-b_true)/b_true)+ torch.abs((a-a_true)/a_true)
        elbo = calc_elbo(a,b,model,meas)
        print(elbo)
        print(ab_error)
        #mean = mean + ab_error.cpu().data.numpy()
        #elbo_mean += elbo.cpu().data.numpy()
        dist_list[q] += (ab_error/no_runs)
        elbo_list[q] += (elbo/no_runs)



# plot graphs and print elbos 
print(dist_list)
print(elbo_list)
plt.figure()
plt.plot(no_meas,dist_list)
plt.xlabel("number of measurements", fontsize = 16)
plt.ylabel("distance to true a and b", fontsize = 16)
plt.savefig('reverse_error_plot')
plt.figure()
plt.plot(no_meas,elbo_list)
