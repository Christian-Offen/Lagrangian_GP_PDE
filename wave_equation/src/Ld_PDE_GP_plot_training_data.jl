#   Learning of Lagrangian as gamblets using a certified kernel-based method
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

using ForwardDiff: gradient, hessian, derivative, jacobian
using Plots
using LinearAlgebra
using Statistics: median
#using BlockArrays
#using SpecialFunctions
using Dates
using FileIO
using FFTW
using Random
using LaTeXStrings


include("7ptStencilFun.jl")
include("TrainingData.jl")
include("Ld_GP_utilities.jl");

rng = MersenneTwister(1234);

mNow() = string(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"));
println("Start: "*mNow())

#   Reference system / true system
#   ================================

### the discretised system is considered the true system

NoSamples=2 # number of solutions to PDE used for training

dimU = 1

## time-spacial domain
l = 1. # length of spatial domain including (periodic) boundary
T = .5 # final time

# discretisation parameters
M = 20 # spatial grid points 
N = 20 # time steps
dx = l/M # periodic boundary conditions
dt = T/N
XMesh   = 0:dx:(M-1)*dx # periodic mesh
XMeshbd = 0:dx:M*dx # periodic mesh
TMesh = 0:dt:N*dt       

# Lagrangian 
Potential(u) = 1/2  * u^2
Lagrangian(u,ut,ux) = 1/2*ut^2-1/2*ux^2-Potential(u)
Ld_ref(u,udown,uleft) = Lagrangian(u,(u-udown)/dt,(u-uleft)/dx)
Ld_ref(x0) = Ld_ref(x0[1],x0[2],x0[3])

println("Compute training data")

training_dataMatrix,training_dataU = CreateTrainingData(Lagrangian; NoSamples=NoSamples,l = l,T = T, M = M, N = N, rng=rng);

problem_size=size(training_dataMatrix,1)+dimU+1
print("Size linear problem "*string(problem_size)*"x"*string(problem_size))

plot_size=(600,400)

# function for plotting
function plotU(U)
    n = size(U)[1]-1 
    TMesh = 0:dt:n*dt
    UPlot = [U U[:,1]] # add repeated boundary for plotting 
    return plot(XMeshbd,TMesh,UPlot,st=:surface,xlabel=L"x",ylabel=L"t",legend=:none,size=plot_size,dpi=600,xticks=3,yticks=3) 
end 

function contourU(U) 
    n = size(U)[1]-1 
    TMesh = 0:dt:n*dt 
    UPlot = [U U[:,1]] # add repeated boundary for plotting 
    return contour(XMeshbd,TMesh,UPlot,xlabel=L"x",ylabel=L"t",legend=:none,size=plot_size,dpi=600) 
end

function heatU(U) 
    n = size(U)[1]-1 
    TMesh = 0:dt:n*dt 
    UPlot = [U U[:,1]] # add repeated boundary for plotting 
    return heatmap(XMeshbd,TMesh,UPlot,xlabel=L"x",ylabel=L"t",legend=:none,size=plot_size,dpi=600) 
end

nofreq=length(rfftfreq(M,M)) 
function plotModesU(U) 
    amplitudes=sum(abs.(mapslices(rfft,U,dims=2)),dims=1)/size(U,1) 
    return bar(0:nofreq-1,amplitudes[:],yscale=:log10,yaxis="amplitude",xaxis="mode number",legend=:none,size=plot_size,dpi=600,xticks=(0:nofreq-1,0:nofreq-1)) 
end 

function plotModesTraining(dataSet) 
    amplitudes = zeros(1,nofreq) 
    for k=1:length(dataSet) 
        amplitudes = amplitudes + sum(abs.(mapslices(rfft,dataSet[k],dims=2)),dims=1)/size(dataSet[k],1) 
    end 
    amplitudes = amplitudes/length(dataSet) 
    return bar(0:nofreq-1,amplitudes[:],yscale=:log10,yaxis="amplitude",xaxis="mode number",legend=:none,size=plot_size,dpi=600,xticks=(0:nofreq-1,0:nofreq-1)) 
end 

for j=1:NoSamples
	Uref0 = training_dataU[:,:,j]
	pRef0=plotU(Uref0)
	savefig(pRef0,"plots/training_data_"*string(j)*".png")
end


println("end of script "*mNow())

#using NBInclude
#nbexport("Ld_GP.jl", "L_Learning_CertifiedGP.ipynb")
