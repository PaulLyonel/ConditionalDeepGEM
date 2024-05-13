import torch
import numpy as np
from torch import nn
from FrEIA.framework import InputNode, OutputNode, Node, GraphINN, ConditionNode
from FrEIA.modules import GLOWCouplingBlock
import FrEIA.modules as Fm
from tqdm import tqdm
torch.set_default_dtype(torch.float32)
import matplotlib.pyplot as plt

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
x_total_y = 2000
n_iter_max = 120
# set log posterior function for mixed noise error model
def get_log_posterior(samples, forward_model, a, b, ys,lambd_bd=10):
    relu=torch.nn.ReLU()
    forward_samps=forward_model(samples)
    prefactor = ((b*forward_samps)**2+a**2)
    p = .5*torch.sum(torch.log(prefactor), dim = 1)
    p2 =  0.5*torch.sum((ys-forward_samps)**2/prefactor, dim = 1)
    p3 = lambd_bd*torch.sum(relu(samples-1)+relu(-samples-1), dim = 1)
    #print(p3)
    return p+p2+p3

# create normalizing flow
def create_INN(num_layers, sub_net_size):
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size,  c_out))

    nodes = [InputNode(DIMENSION, name='input')]
    cond = ConditionNode(ndim_y, name='condition')
    for k in range(num_layers):
        nodes.append(Node(nodes[-1],
                          GLOWCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':1.4},
                          conditions = cond,
                          name=F'coupling_{k}'))

    nodes.append(OutputNode(nodes[-1], name='output'))
    model = GraphINN(nodes + [cond], verbose=False).to(device)
    return model
    
# train with forward KL
def train_inn_epoch(model,optimizer, a, b, forward_model, batch_size = 512):
    for k in range(10):
        x = torch.rand(batch_size, DIMENSION, device = device)*2-1
        ys = forward_model(x) + b*forward_model(x)*torch.randn(batch_size, ndim_y, device = device)+a*torch.randn(batch_size, ndim_y, device = device)
        optimizer.zero_grad()
        out = model(x, c = ys, rev = True)
        output = out[0].to(device)
        jac = out[1].to(device)
        l = 0.5*torch.sum(output**2)/batch_size
        l = l - torch.mean(jac)
        l.backward()
        optimizer.step()
    return l

def calc_elbo(a,b,model, meas):
    with torch.no_grad():
        ys = torch.zeros(len(meas), ndim_y, device = device)
        ys += meas.to(device)
        ys = ys.repeat(2000//len(meas),1)
        z = torch.randn(len(ys), DIMENSION, device = device)
        out,jac = model(z, c = ys)
        elbo = -get_log_posterior(out, forward_model, a,b, ys).mean() + jac.mean()
    return elbo
    


def savePost(posterior, no_params, x_true, l):
    fig, axes = plt.subplots(figsize=[9,9], nrows=no_params, ncols=no_params);
    posterior = posterior[0].cpu().data.numpy()
    x_true = x_true.cpu().data.numpy().reshape(1,DIMENSION)
    for j in range(no_params):
        for k in range(no_params):
            axes[j,k].get_xaxis().set_ticks([]);
            axes[j,k].get_yaxis().set_ticks([]);
            #if j == len(params)-1: axes[j,k].set_xlabel(k);
            if j == k:
                axes[k,k].hist(posterior[:,j], bins=50,  alpha=0.3,range = (-1.2,1.2));
                axes[k,k].hist(posterior[:,j], bins=50,  histtype="step", range = (-1.2,1.2));
                axes[k,k].axvline(x_true[:,j])
            else:
                val, x, y = np.histogram2d(posterior[:,j], posterior[:,k], bins=100, range = [[-1.2, 1.2], [-1.2, 1.2]]);
                axes[j,k].contourf(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=1);
    plt.tight_layout()

    plt.savefig('grid_forw' +str(l)+'.png')
    plt.close()

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
       
        max_elbo = 0.
        opt_a = 0.
        opt_b = 0.
        a_l = torch.linspace(0.001,0.03, 8)
        b_l = torch.linspace(0.01,0.2, 8)
        count = 0.
        with tqdm(range(64), desc='elbo') as tq:
            for a in a_l:
                for b in b_l:
                    model = create_INN(4,256)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                    for k in range(n_iter_max):
                        l = train_inn_epoch(model, optimizer, a.detach(), b.detach(),forward_model)

                        with torch.no_grad():
                            elbo = calc_elbo(a,b,model,meas)
                            if elbo > max_elbo:
                                max_elbo = elbo
                                torch.save(model.state_dict(), 'grid_KL_model.pt')
                                opt_a = a.clone()
                                opt_b = b.clone()
                        tq.set_description(f"elbo {elbo.item(), max_elbo}")
                    count += 1
                    tq.update(1)


        model.load_state_dict(torch.load('grid_KL_model.pt'))   
        a = opt_a.clone()
        b = opt_b.clone()
        y_meas = torch.zeros(2000,ndim_y,device = device)
        y_meas = y_meas + meas[0]
        z = torch.randn(2000,DIMENSION, device = device)
        samples = model(z,c = y_meas)
        savePost(samples, DIMENSION, x_true[0], p*len(no_meas)+q)
        ab_error = torch.abs((b-b_true)/b_true)+ torch.abs((a-a_true)/a_true)
        elbo = calc_elbo(a,b,model,meas)
        print(elbo)
        print(ab_error)
        #mean = mean + ab_error.cpu().data.numpy()
        #elbo_mean += elbo
        dist_list[q] += (ab_error/no_runs)
        elbo_list[q] +=(elbo/no_runs)

print(dist_list)
print(elbo_list)



