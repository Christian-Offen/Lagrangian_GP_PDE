#   Machine learning of Schrödinger equation using CertifiedGP method
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

#   load packages, scripts
#   ======================

using Random
using ForwardDiff
using NLsolve
using LinearAlgebra
using FFTW
using Dates
using FileIO
using Plots
using LaTeXStrings
using ProgressMeter

include("src/Ld_GP_utilities.jl");
include("src/9ptStencil.jl")
include("src/DELSolver9ptStencil.jl")
include("src/TrainingData.jl")
include("src/plotting_tools.jl");
include("src/PredictComparePlot.jl")


rng = MersenneTwister(4321); # init random state
mNow()=Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")

println("Start "*mNow())

#   define true system and compute training data
#   ============================================

NoSamples=30;	# number of solutions to PDE used for training

## time-spacial domain
l = 1. # length of spatial domain including (periodic) boundary
T = .14 # final time

# discretisation parameters
M =  10 # spatial grid points 
N =  8 # time steps
dim = 2 # real dimension of target space of solution to PDE
dx = l/M # periodic boundary conditions
dt = T/N
XMesh   = 0:dx:(M-1)*dx # periodic mesh
XMeshbd = 0:dx:M*dx # periodic mesh
TMesh = 0:dt:N*dt       
dimU = 2     # real two-dimensional

# Lagrangian
V(r) = r
alpha(u) = [u[2];-u[1]]
H(u,ux) = sum(ux.^2) + V(sum(u.^2))
L_ref(u,ut,ux) = dot(alpha(u),ut)-H(u,ux)
Ld_ref,Ldx_ref,Ldxd_ref,firstStep_ref = DiscretiseLDensity(L_ref)
function Ld_refInstance(u,uup,uright,uupright); return Ld_ref(dt,dx,u,uup,uright,uupright); end
function Ld_refInstance(uu::Vector); return Ld_refInstance(uu[1:2],uu[3:4],uu[5:6],uu[7:8]); end

# define solver functions
DELSolve_ref, DELdx_ref = DELSolver9ptStencil(Ld_refInstance)

println("\nCompute training data "*mNow()); flush(stdout);
training_dataStencils, training_dataU = CreateTrainingData(L_ref, rng; NoSamples=NoSamples,l = l,T = T, M = M, N = N);

save("data/TrainingData_"*mNow()*".jld2","training_dataStencils",training_dataStencils,"training_dataU",training_dataU)

problem_size=size(training_dataStencils,1)*dimU+dimU+1
println("Size of linear problem "*string(problem_size)*"x"*string(problem_size)); flush(stdout);

#   plot training sample
#   =================

# instantiate plot functions
plotArgs3d = Dict(:xlabel=>L"x", :ylabel=>L"t", :legend=>:none, :dpi=>300)
plotArgsC = Dict(:xlabel=>L"x", :ylabel=>L"t", :dpi=>300)
plotU, contourU = InstantiatePlotFun(dt,dx, plotArgs3d=plotArgs3d,plotArgsC=plotArgsC); 

training_sampleNo = 2

# plot real/imaginary 
ptrainingRe = plotU(transpose(training_dataU[1,:,:,training_sampleNo]))
#plot!(title="training sample, real", dpi=300)
ptrainingIm = plotU(transpose(training_dataU[2,:,:,training_sampleNo]))
#plot!(title="training sample, im")
pTrainingSample=plot(ptrainingRe,ptrainingIm,layout=(1,2))

savefig(ptrainingRe,"plots/ptrainingRe_"*mNow()*".png")
savefig(ptrainingIm,"plots/ptrainingIm_"*mNow()*".png")

#   kernel definition
#   =================

# kernel definition
lengthscale = ones(4*dimU); #[0.9; 0.9; 0.9; 0.9]

# squared exponential
kernel(a,b) = exp(-0.5*sum(((a-b)./lengthscale).^2))

#   Computation of gamblet
#   ======================

# normalisation
baseStencil = zeros(9,2);
normalPpt_base = ones(dimU+1);

Ld_ml, dEL_ml, var_operator, var_pts, var_el, _ ,Theta_fact=Lagrangian_Gamblet(kernel,training_dataStencils,baseStencil;normalisation_balancing=normalPpt_base);

# define solver functions
DELSolve_ml, DELdx_ml = DELSolver9ptStencil(Ld_ml)

println("\nComputation of posterior process completed "*mNow()*"\n"); flush(stdout);

#   Tests learning success
#   ======================

# check normalisation
println("Normalisation: "); flush(stdout);
println("Ld_ml(basept) "*string(Ld_ml(StencilToQuad(baseStencil)))*", conj_local_ml(baseStencil) "*string(conjp_local(x->[Ld_ml(x)],baseStencil))); flush(stdout);

# Test how well collocation conditions are fulfilled
collocation_test0 = mapslices(dEL_ml,training_dataStencils,dims=(2,3))
mCollTest=maximum(abs.(collocation_test0))         # max error EL

println("Max el(Ld_ml,data) "*string(mCollTest)); flush(stdout);

# check whether variance of EulerLagrange operator (interpreted as observable) is zero at data
var_el_val = mapslices(var_el,training_dataStencils,dims=(2,3))
max_var_ml=maximum(abs.(var_el_val))
println("Max var_el(Ld_ml,data) "*string(max_var_ml)); flush(stdout);

## high memory requirement - only in debugging
# conditioning of linear problem in gamblet evaluation
#println("Condition linear problem in gamblet evaluation "*string(cond(Theta))); flush(stdout);
#
# symmetry
#println("Symmetry fail of theta "*string(maximum(abs.(Theta-Theta')))); flush(stdout);   
#
# eigenvalues
#eigsTheta=eigvals(Theta);
#println("Smallest eigenvalue of theta: "*string(minimum(real.(eigsTheta)))); flush(stdout);
#
#indexPos=argmax(real(eigsTheta).>0.) # index of first positive eigenvalue
#println(string(indexPos-1)*" numerically negative eigenvalues"); flush(stdout);
#
#pEIgsTheta = plot(real.(eigsTheta[indexPos:end]),yaxis=:log, label="positive eigenvalues Theta"); flush(stdout);
#
#savefig(pEIgsTheta,"plots/eigTheta"*mNow()*".pdf")

#   recovery of training data
#   =========================

println("\nRecovery training data tests "*mNow()); flush(stdout);

u0 = training_dataU[:,:,1,1]
u1 = training_dataU[:,:,2,1];

try
	U_ml,U_ref,stnd_dev, pCollect, errs = PredictCompare(DELSolve_ml,DELSolve_ref, dt,dx, u0,u1,N,var_el; prefix="RecoverTrainingData_", postfix="_"*mNow());
	println("Training data recovery max/l2 error "*string(errs)); flush(stdout);
catch	e
	println("FAIL "*mNow());
	println(e)
	flush(stdout);
end



#   predict new solution
#   ====================

println("\nPredict solution unseen initial data "*mNow()); flush(stdout);
NSteps2 = N

# sample new data and use as test
_, testData = CreateTrainingData(L_ref, rng; NoSamples=1,l = l,T = T, M = M, N = N) # unseen random sample
u0 = testData[:,:,1,1]
u1 = testData[:,:,2,1]

try
	U2_ml,U2_ref,stnd_dev2, pCollect2, errs2 = PredictCompare(DELSolve_ml,DELSolve_ref, dt,dx, u0,u1,NSteps2,var_el; prefix="predict_unseen_", postfix="_"*mNow());
	println("max/l2 error "*string(errs2)); flush(stdout);
catch	e
	println("FAIL "*mNow());
	println(e)
	flush(stdout);
end


#   Travelling wave
#   ===============

#   define true TW and consistency checks
#   –––––––––––––––––––––––––––––––––––––

println("\nTravelling wave test "*mNow()); flush(stdout);

# define true travelling wave of discrete system
wave_no = 1

beta = 2*pi*wave_no/l   # imaginary unit not included

Vd = 1
hbar = 1
c_tw_discrete = 2/(beta*dt)*atan(2/hbar*dt/dx^2*tan(1/2*beta*dx)^2+Vd*dt/(2*hbar))     #+ 2/(alpha*dt)*s*pi, s \in \Z

amplitude_discrete = 0.0 + 1.0*im
tw_discrete(t,x) = amplitude_discrete*exp(im*beta*(x-c_tw_discrete*t))
Psi_tw_discrete = tw_discrete.(TMesh,transpose(XMesh))
U_tw_discrete = cat(real.(Psi_tw_discrete),imag.(Psi_tw_discrete),dims=3)
U_tw_discrete = permutedims(U_tw_discrete,[3,2,1])

pTW_re_true = plotU(real.(Psi_tw_discrete))
plot!(title=L"\mathrm{re}(U_\mathrm{TW})")
pTW_im_true = plotU(real.(Psi_tw_discrete))
plot!(title=L"\mathrm{im}(U_\mathrm{TW})")

#plot(pTW_re_true,pTW_im_true)

savefig(pTW_re_true,"plots/pTW_re_true_"*mNow()*".pdf")
savefig(pTW_im_true,"plots/pTW_im_true_"*mNow()*".pdf")

cont_TW_re_true = contourU(real.(Psi_tw_discrete))
plot!(title=L"\mathrm{re}(U_\mathrm{TW})")
cont_TW_im_true = contourU(real.(Psi_tw_discrete))
plot!(title=L"\mathrm{im}(U_\mathrm{TW})")

#plot(cont_TW_re_true,cont_TW_im_true)

savefig(cont_TW_re_true,"plots/cont_TW_re_true_"*mNow()*".pdf")
savefig(cont_TW_im_true,"plots/cont_TW_im_true_"*mNow()*".pdf")

# Check how well true TW fulfills DEL(L_ml)

dEL_ml_trueTW = dropdims(mapslices(dEL_ml,CollectStencils(U_tw_discrete),dims=(2,3)),dims=2);
dEL_ml_trueTW_err = maximum(abs.(dEL_ml_trueTW))
println("true TW fulfills DEL(L_ml) with max error of "*string(dEL_ml_trueTW_err)); flush(stdout);

#   use TW as initial data
#   ––––––––––––––––––––––

# use TW as initial data

NTW = ceil(Int64,N/2)
u0TW = U_tw_discrete[:,:,1]
u1TW = U_tw_discrete[:,:,2];

try
	UTW_ml,UTW_ref,stnd_devTW, pCollectTW, errsTW = PredictCompare(DELSolve_ml,DELSolve_ref, dt,dx, u0TW,u1TW,NTW,var_el; prefix="predict_TW_", postfix="_"*mNow());

	println("Consistency TW with true model (sanity check): "*string(maximum(abs.(UTW_ref - U_tw_discrete[:,:,1:(NTW+1)])))); flush(stdout); # Sanity check
	println("max/l2 error "*string(errsTW)); flush(stdout);
catch	e
	println("FAIL "*mNow());
	println(e)
	flush(stdout);
end


#   use TW as initial data long term prediction
#   –––––––––––––––––––––––––––––––––––––––––––

println("\nTravelling wave test long term prediction "*mNow()); flush(stdout);

NTW = 2*N;
try
	UTW_ml,UTW_ref,stnd_devTW, pCollectTW, errsTW = PredictCompare(DELSolve_ml,DELSolve_ref, dt,dx, u0TW,u1TW,NTW,var_el; prefix="predict_TW_long_", postfix="_"*mNow());

	println("max/l2 error "*string(errsTW)); flush(stdout);
catch	e
	println("FAIL "*mNow());
	println(e)
	flush(stdout);
end
	

println("End of script "*mNow())
