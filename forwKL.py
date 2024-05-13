import torch
import numpy as np
from torch import nn
from FrEIA.framework import InputNode, OutputNode, Node, GraphINN, ConditionNode
from FrEIA.modules import GLOWCouplingBlock
import FrEIA.modules as Fm
from tqdm import tqdm
# use float 64 because forward model close to zero sometimes 
torch.set_default_dtype(torch.float64)
import matplotlib.pyplot as plt
from utils_high import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set problem parameters
DIMENSION = 7
ndim_y = 77
n_iter_max = 5000

# load forward model
forward_model = nn.Sequential(nn.Linear(DIMENSION, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256,  ndim_y)).to(device)

forward_model.load_state_dict(torch.load('forw_new.pt'))
for param in forward_model.parameters():
    param.requires_grad=False

    

# define hyperparameters

# number of measurements and number of runs for evaluation
no_meas = [1,2,4,8]
no_runs = 5
dist_list = np.zeros((4))
elbo_list = np.zeros((4))
a_true = 0.03
b_true = 0.25

for p in range(no_runs):

    print(p)
    mean = 0
    elbo_mean = 0
    torch.manual_seed(p)
    x_true = torch.rand(8, DIMENSION, device = device)
    meas2 = forward_model(x_true)+b_true*torch.randn_like(forward_model(x_true), device = device)*forward_model(x_true)+a_true*torch.randn_like(forward_model(x_true), device = device)
    for q in range(len(no_meas)):
        meas = meas2[:no_meas[q]:,]

        model = create_INN(4,256)
        a = torch.ones(1,1, device = device).clone().detach()*0.3
        b = torch.ones(1,1, device = device).clone().detach()*0.3
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        max_elbo = 0.
        opt_a = 0.
        opt_b = 0.
        tq = tqdm(range(n_iter_max), desc='elbo')
        for it in tq:
            l,a,b = train_inn_epoch_forw(model, meas, optimizer, a.detach(), b.detach(),forward_model)
            with torch.no_grad():
                elbo = calc_elbo(a,b,model,meas)
                if elbo > max_elbo:
                    max_elbo = elbo
                    torch.save(model.state_dict(), 'for_KL_model.pt')
                    opt_a = a.clone()
                    opt_b = b.clone()
            tq.set_description(f"elbo {elbo.item(),max_elbo}")

        model.load_state_dict(torch.load('for_KL_model.pt'))   
        a = opt_a.clone()
        b = opt_b.clone()
        y_meas = torch.zeros(2000,ndim_y,device = device)
        y_meas = y_meas + meas[q]
        z = torch.randn(2000,7, device = device)
        samples = model(z,c = y_meas)

        savePost(samples, DIMENSION, x_true[q], p*len(no_meas)+q, 'posterior_forw')

        ab_error = torch.abs((b-b_true)/b_true)+ torch.abs((a-a_true)/a_true)
        elbo = calc_elbo(a,b,model,meas)

        print(elbo)
        print(ab_error)
       
        dist_list[q] += (ab_error/no_runs)
        elbo_list[q] +=(elbo/no_runs)
        
print(dist_list)
print(elbo_list)

plt.figure()
plt.plot(no_meas,dist_list)
plt.xlabel("number of measurements", fontsize = 16)
plt.ylabel("distance to true a and b", fontsize = 16)
plt.savefig('forward_error_plot')





