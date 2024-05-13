import torch
import numpy as np
from torch import nn
from FrEIA.framework import InputNode, OutputNode, Node, GraphINN, ConditionNode
from FrEIA.modules import GLOWCouplingBlock
import FrEIA.modules as Fm
# use float 64 because forward model close to zero sometimes 
torch.set_default_dtype(torch.float64)
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set problem parameters
DIMENSION = 7
ndim_y = 77

relu = torch.nn.ReLU()
x_total_y = 2000
# stabilizing epsilon in em step to avoid nans
stab_eps = 1e-10

forward_model = nn.Sequential(nn.Linear(DIMENSION, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256,  ndim_y)).to(device)

forward_model.load_state_dict(torch.load('forw_new.pt'))
for param in forward_model.parameters():
    param.requires_grad=False

# set log posterior function for mixed noise error model
def get_log_posterior(samples, forward_model, a, b, ys,lambd_bd=10):
    relu=torch.nn.ReLU()
    forward_samps=forward_model(samples)
    prefactor = ((b*forward_samps)**2+a**2)
    p = .5*torch.sum(torch.log(prefactor), dim = 1)
    p2 =  0.5*torch.sum((ys-forward_samps)**2/prefactor, dim = 1)
    p3 = lambd_bd*torch.sum(relu(samples-1)+relu(-samples), dim = 1)
    #print(p3)
    return p+p2+p3

# create normalizing flow, cf. FreiA package
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

# em algorithm with 20 default steps
def em_ab(a_sq_init,b_sq_init,f_x,y,steps=20):
    def compute_c1_c2(a_sq_r,b_sq_r,f_x_inp,y_inp):
        l2_dist=(y_inp-f_x_inp)**2
        c_nenner=a_sq_r+b_sq_r*f_x_inp**2
        c_nenner_sq=c_nenner**2
        v_variance=a_sq_r*b_sq_r*f_x_inp**2/c_nenner
        
        c1_=-(l2_dist*(1-2*a_sq_r/c_nenner+a_sq_r**2/c_nenner_sq)+v_variance)/(torch.max(f_x_inp**2, torch.ones_like(f_x_inp**2)*stab_eps))
        c2_=-(l2_dist*a_sq_r**2/c_nenner_sq+v_variance)
        c1=torch.sum(c1_/y.shape[1])
        c2=torch.sum(c2_/y.shape[1])
        return c1,c2

    a_sq = a_sq_init
    b_sq = b_sq_init
    for step in range(steps):
        c1,c2=compute_c1_c2(a_sq,b_sq,f_x,y)
        a_sq_new=-c2/(y.shape[0])
        b_sq_new=-c1/(y.shape[0])

        a_sq=a_sq_new
        b_sq=b_sq_new
    return a_sq,b_sq
    
# train with forward KL
def train_inn_epoch_forw(model, meas,optimizer, a, b, forward_model, batch_size = 512):
    for k in range(10):
        x = torch.rand(batch_size, DIMENSION, device = device)
        ys = forward_model(x) + b*forward_model(x)*torch.randn(batch_size, ndim_y, device = device)+a*torch.randn(batch_size, ndim_y, device = device)
        optimizer.zero_grad()
        out = model(x, c = ys, rev = True)
        output = out[0].to(device)
        jac = out[1].to(device)
        l = 0.5*torch.sum(output**2)/batch_size
        l = l - torch.mean(jac)
        l.backward()
        optimizer.step()
    for k in range(1):
        with torch.no_grad():
            ys = torch.zeros(len(meas), ndim_y, device = device)
            ys += meas.to(device)
            ys = ys.repeat(x_total_y//len(meas),1)

            z = torch.randn(len(ys), DIMENSION, device = device)

            out = model(z, c = ys)
            output = out[0].to(device)
            fx = forward_model(output)
            a,b = em_ab(a**2,b**2,fx,ys)
            a = torch.sqrt(a.abs())
            b = torch.sqrt(b.abs())
    return l,a,b

# train with reverse KL
def train_inn_epoch_rev(model, meas, optimizer, a, b, forward_model, batch_size = 512):
    for k in range(10):
        x = torch.rand(batch_size, DIMENSION, device = device)
        ys = forward_model(x) + b*forward_model(x)*torch.randn(batch_size, ndim_y, device = device)+a*torch.randn(batch_size, ndim_y, device = device)
        z = torch.randn(len(ys), DIMENSION, device = device)
        out,jac = model(z, c = ys)
        optimizer.zero_grad()
        l = get_log_posterior(out, forward_model, a,b, ys).mean() - jac.mean()
        l.backward()
        optimizer.step()
    for k in range(1):
        with torch.no_grad():

            ys = torch.zeros(len(meas), ndim_y, device = device)
            ys += meas.to(device)
            ys = ys.repeat(x_total_y//len(meas),1)

            z = torch.randn(len(ys), DIMENSION, device = device)

            out = model(z, c = ys)
            output = out[0].to(device)
            fx = forward_model(output)
            a,b = em_ab(a**2,b**2,fx,ys)
            a = torch.sqrt(a.abs())
            b = torch.sqrt(b.abs())
    return l,a,b

# calculate elbo
def calc_elbo(a,b,model, meas):
    with torch.no_grad():
        ys = torch.zeros(len(meas), ndim_y, device = device)
        ys += meas.to(device)
        ys = ys.repeat(10000//len(meas),1)
        z = torch.randn(len(ys), DIMENSION, device = device)
        out,jac = model(z, c = ys)
        elbo = -get_log_posterior(out, forward_model, a,b, ys).mean() + jac.mean()
    return elbo
    

# plotting function
def savePost(posterior, no_params, x_true, l, name):
    fig, axes = plt.subplots(figsize=[9,9], nrows=no_params, ncols=no_params);
    posterior = posterior[0].cpu().data.numpy()
    x_true = x_true.cpu().data.numpy().reshape(1,7)
    for j in range(no_params):
        for k in range(no_params):
            axes[j,k].get_xaxis().set_ticks([]);
            axes[j,k].get_yaxis().set_ticks([]);
            #if j == len(params)-1: axes[j,k].set_xlabel(k);
            if j == k:
                axes[k,k].hist(posterior[:,j], bins=50,  alpha=0.3,range = (0.,1.2));
                axes[k,k].hist(posterior[:,j], bins=50,  histtype="step", range = (0.,1.2));
                axes[k,k].axvline(x_true[:,j])
            else:
                val, x, y = np.histogram2d(posterior[:,j], posterior[:,k], bins=100, range = [[0., 1.2], [0., 1.2]]);
                axes[j,k].contourf(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=1);
    plt.tight_layout()
    plt.savefig(name+str(l)+'.png')
    plt.close()

