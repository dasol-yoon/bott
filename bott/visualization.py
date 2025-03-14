# Visualization functions for the package
import torch
import matplotlib.pyplot as plt
from .metrics import pixelSSE

from botorch.sampling.normal import IIDNormalSampler #import package

# TODO
# Need some major revision of this module
# We probably need a model object to pass the required fields into each plotting functions

tkwargs = {"dtype":torch.double}


# Plot the posterior for a single simulation output y
def plot_posterior_classic(ax, model, gridX1, gridX2, train_x, train_y, legend=True):
    with torch.no_grad(): # no need for gradients
        test_x = torch.stack([gridX1.flatten(),gridX2.flatten()],dim=1)
        # compute posterior
        posterior_x_domain = model.posterior(test_x.to(**tkwargs))

        inner_sample = IIDNormalSampler(sample_shape=torch.Size([2048])) #define based sample
        samples = inner_sample(posterior_x_domain) # do the sample
        
        samples_obj = -pixelSSE(samples) #DY: make it objective only
        
        samplesmedian,_ = samples_obj.median(dim=0) # compute median (you can use mean)
        samples025 = samples_obj.quantile(q=0.25,dim=0) #compute quantile
        samples0975=samples_obj.quantile(q=0.975,dim=0) #compute quantile
         
        # Plot training points as black stars
        ax.plot(train_x[:, 0].cpu().numpy(), 
                train_x[:, 1].cpu().numpy(), 
                -pixelSSE(train_y).cpu().numpy().squeeze(),'k*')

        # Plot posterior means as a surface 
        post = ax.plot_surface(gridX1, gridX2, 
                               samplesmedian.cpu().reshape(gridX1.shape), #.detach() is not changing the device
                               cmap='autumn_r', alpha=0.5)
        
        # Upper and lower bounds
        ax.plot_wireframe(gridX1,gridX2,
                          samples025.cpu().reshape(gridX1.shape),
                          alpha=0.5) #lw=0.5, rstride=2, cstride=2,
        ax.plot_wireframe(gridX1,gridX2,
                          samples0975.cpu().reshape(gridX1.shape),alpha=0.5)
        ax.set_xlabel('thickness');ax.set_ylabel('tilt')
        
    if legend:
        plt.legend(['Observed Data', 'Mean', 'Credible Interval'])
        

# Plot the posterior for a single simulation output y
def plot_posterior_composite(ax, model, gridX1, gridX2, train_x, train_y, legend=True):
    with torch.no_grad(): # no need for gradients
        test_x = torch.stack([gridX1.flatten(),gridX2.flatten()],dim=1)
        # compute posterior
        posterior_x_domain = model.posterior(test_x.to(**tkwargs))

        inner_sample = IIDNormalSampler(sample_shape=torch.Size([2048])) #define based sample
        samples = inner_sample(posterior_x_domain) # do the sample
        
        samples_obj = g(samples) #DY: make it objective only
        
        samplesmedian,_ = samples_obj.median(dim=0) # compute median (you can use mean)
        samples025 = samples_obj.quantile(q=0.25,dim=0) #compute quantile
        samples0975=samples_obj.quantile(q=0.975,dim=0) #compute quantile
         
        # Plot training points as black stars
        ax.plot(train_x[:, 0].cpu().numpy(), 
                train_x[:, 1].cpu().numpy(), 
                g(train_y).cpu().numpy().squeeze(),'k*')

        # Plot posterior means as a surface 
        post = ax.plot_surface(gridX1, gridX2, 
                               samplesmedian.cpu().reshape(gridX1.shape), #.detach() is not changing the device
                               cmap='autumn_r', alpha=0.5)
        
        # Upper and lower bounds
        ax.plot_wireframe(gridX1,gridX2,
                          samples025.cpu().reshape(gridX1.shape),
                          alpha=0.5) #lw=0.5, rstride=2, cstride=2,
        ax.plot_wireframe(gridX1,gridX2,
                          samples0975.cpu().reshape(gridX1.shape),alpha=0.5)
        ax.set_xlabel('thickness');ax.set_ylabel('tilt')
        
    if legend:
        plt.legend(['Observed Data', 'Mean', 'Credible Interval'])

#x_domain = torch.linspace(lower, upper, 1000).unsqueeze(-1).to(torch.double)
def plot_model_EI(ax, gridX1, gridX2, model,acqf,iter_no):
    test_x = torch.stack([gridX1.flatten(),gridX2.flatten()],dim=1)

    post_at_x_domain = model.posterior(test_x)
    mean_at_x_domain = post_at_x_domain.mean
    std_at_x_domain = torch.sqrt(post_at_x_domain.variance)
    lo_at_x_domain = mean_at_x_domain-1.96*std_at_x_domain
    up_at_x_domain = mean_at_x_domain+1.96*std_at_x_domain
    
    acqf_val = acqf(x_domain.unsqueeze(-1))
    axis[5,1].set_title(plot_name[count_idx])
    axis[5,1].plot(x_domain,acqf_val.detach().numpy(),color='green',label='EI')
    axis[5,1].plot(new_point.detach().numpy(),new_point_EI.detach().numpy(),marker='*',linestyle='none', markersize=10, color='yellow',label='Candidate')
    axis[5,1].legend()
    plt.suptitle(f"Iteration {iter_no}", fontsize=14, y=0.91)  # Reduce `y` to bring closer
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplot grid to fit universal title
    plt.show()
    
def plotEI(ax,gridX1,gridX2, ei, new_pt):
    test_x = torch.stack([gridX1.flatten(),gridX2.flatten()],dim=1)
    acq = ei.forward(test_x.unsqueeze(1).to(**tkwargs))
    acqValues = acq.detach().cpu().numpy().reshape(gridX1.shape)
    
    # Contour plot of EI over the 2D input space
    ct = ax.contourf(gridX1.numpy(), gridX2.numpy(), 
                     acqValues, cmap='viridis')
    plt.colorbar(ct)
    
    # Mark the new point found by the optimizer
    ax.plot(new_pt[0,0].cpu().numpy(), new_pt[0,1].cpu().numpy(), 'ro')
    
    ax.set_title('Expected Improvement (EI)')
    ax.set_xlabel('Thickness'); ax.set_ylabel('Tilt');