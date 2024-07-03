#   Learning of Lagrangian as posterior GPs
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

using ForwardDiff: gradient, hessian, derivative, jacobian
using Plots
using LinearAlgebra
using NLsolve
using ProgressMeter
using Dates
using FileIO
using FFTW
using Random
using LaTeXStrings


include("src/7ptStencilFun.jl")
include("src/TrainingData.jl")
include("src/Ld_GP_utilities.jl")
include("src/plotting_tools.jl");

rng = MersenneTwister(1234);

mNow() = string(Dates.format(now(), "yyyy-mm-dd_HH:MM:SS"));
println("Start: "*mNow()*"\n\n")

#   Reference system / true system
#   ==============================

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
println("Size linear problem "*string(problem_size)*"x"*string(problem_size))

# instantiate plot functions
plotArgs3d = Dict(:xlabel=>L"x", :ylabel=>L"t", :legend=>:none, :dpi=>400, :size=>(300,300))
plotArgsC = Dict(:xlabel=>L"x", :ylabel=>L"t", :dpi=>400, :size=>(300,300))
plotU, contourU,heatU, contourU!, plotAllU, plotModesU,plotModesTraining = InstantiatePlotFun(dt,dx, plotArgs3d=plotArgs3d,plotArgsC=plotArgsC); 

# plot training data
for j=1:NoSamples
	Uref0 = training_dataU[:,:,j]
	pRef0=plotU(Uref0)
	savefig(pRef0,"plots/training_data_"*string(j)*".png")
end


#   kernel definition
#   =================

# kernel definition
lengthscale = 3*ones(dimU); #[0.9; 0.9; 0.9; 0.9]

# squared exponential
kernel(a,b) = exp(-0.5*sum(((a-b)./lengthscale).^2))
kernel(q1,q1dot,q2,q2dot) = kernel([q1;q1dot],[q2;q2dot])

#   Computation of posterior process
#   ================================

println("Define posterior process Ld and variance operators "*mNow())

baseStencil = zeros(7);
normalPpt_base = ones(dimU+1);

Ld_ml, dEL_ml, var_operator, var_pts, var_P, var_el, Theta,Theta_fact=Lagrangian_Posterior(kernel,training_dataMatrix,baseStencil;normalisation_balancing=normalPpt_base);

println(mNow())

#   Consistancy tests learned model
#   ===============================

println("\n \n Consistency tests for learned model \n")

# check normalisation
println("Normalisation: ")
println("Ld_ml(basept) "*string(Ld_ml(baseStencil[1:3]))*", conj_local_ml(baseStencil) "*string(conjp_local(x->[Ld_ml(x)],baseStencil)))

# check whether variance of EulerLagrange operator (interpreted as observable) is zero at data
var_el_val = var_el(training_dataMatrix)
max_var_ml=maximum(var_el_val)
println("Max var_el(L_ml,data) "*string(max_var_ml))

# Test how well collocation conditions are fulfilled
collocation_test0 = dEL_ml(training_dataMatrix)
mCollTest=maximum(abs.(collocation_test0))         # max error EL

println("Max el(Ld_ml,data) "*string(mCollTest))

# conditioning of linear problem in evaluation
println("Condition linear problem in posterior evaluation "*string(cond(Theta)))

# symmetry
println("Max. symmetry error Theta: "*string(maximum(abs.(Theta-Theta'))))

# eigenvalues
eigsTheta=eigvals(Theta);
println("Smallest eigenvalue Theta: "*string(minimum(real.(eigsTheta))))

indexPos=argmax(real(eigsTheta).>0.) # index of first positive eigenvalue
println(string(indexPos-1)*" numerically negative eigenvalues")

pEIgsTheta = plot(real.(eigsTheta[indexPos:end]),yaxis=:log, label="eig Theta")

savefig(pEIgsTheta,"plots/numericalEigenvaluesTheta.pdf")

#   recovery of training data
#   =========================

println("\n \n recovery training data tests "*mNow()*"\n")

U_test = training_dataU[:,:,1]
u0 = U_test[1,:]
u1 = U_test[2,:]
U_ml=PDEContinue(Ld_ml, u0, u1; time_steps = N);

max_err = maximum(abs.(U_ml-U_test))
l2_err  = sqrt(sum((U_ml-U_test).^2)*dx*dt)
println("max./l2 error: "*string(max_err)*" / "*string(l2_err))

pUml =plotU(U_ml)
pUref=plotU(training_dataU[1:size(U_ml,2),:,1]);
pURecoveryTestCompare=plot(pUml,pUref)

savefig(pUml,"plots/RecoveryTest_ml_prediction.pdf")
savefig(pUref,"plots/RecoveryTest_true_motion.pdf")
savefig(pURecoveryTestCompare,"plots/RecoveryTest.pdf")

variance_Uml = var_el(CollectStencils(U_ml))
st_deviation_Uml=sqrt.((x->max(0.,x)).(variance_Uml)); # map numerically negative entries to 0

println("Max variance: "*string(maximum(variance_Uml)))
println("Min variance: "*string(minimum(variance_Uml)))

st_deviation_Uml_spatial = [zeros(2,M)
 reshape(st_deviation_Uml,(size(U_ml,2),size(U_ml,1)-2))'];

pStdUmlRecoveryTest=heatU(st_deviation_Uml_spatial)

savefig(pStdUmlRecoveryTest,"plots/RecoveryTest_standardDeviation.pdf")

#   predict new solution
#   ====================

println("\n \n Predict solution to field theory from unseen initial data "*mNow()*"\n")

NSteps2 = 20
u0=-cos.(2*pi*XMesh)
pInitVal_unseen=plot(XMesh,u0,xlabel="x",label="u0")

savefig(pInitVal_unseen,"plots/PredictionTest_InitVal.pdf")

u1 = u0
U_ml2=PDEContinueProgressBar(Ld_ml, u0, u1; time_steps = NSteps2);

U_ref2=PDEContinueProgressBar(Ld_ref, u0, u1; time_steps = NSteps2);

max_err = maximum(abs.(U_ml2-U_ref2))
l2_err  = sqrt(sum((U_ml2-U_ref2).^2)*dx*dt)
println("max./l2 error: "*string(max_err)*" / "*string(l2_err))

pUml2 =plotU(U_ml2)
#plot!(title="predicted")
pUref2=plotU(U_ref2)
#plot!(title="reference")
plot(pUml2,pUref2)

savefig(pUml2,"plots/predicted_evolution.png")
savefig(pUref2,"plots/predicted_evolution_reference.png")

variance_Uml2 = var_el(CollectStencils(U_ml2))
st_deviation_Uml2=sqrt.((x->max(0.,x)).(variance_Uml2)); # map numerically negative entries to 0
st_deviation_Uml2_spatial = [zeros(2,M)
 reshape(st_deviation_Uml2,(size(U_ml2,2),size(U_ml2,1)-2))'];

println("max variance: "*string(maximum(variance_Uml2)))

pUml2Heat=heatU(st_deviation_Uml2_spatial)
plot!(right_margin=40Plots.pt, colorbar=true)

savefig(pUml2Heat,"plots/predicted_evolution_stand_dev_heatmap.pdf")

av_comp_sample_max=50
training_min = minimum(training_dataU)
training_span=maximum(training_dataU)-minimum(training_dataU)
random_stencils = training_span.*rand(rng,av_comp_sample_max,7) .+ training_min;
var_random_samples=var_el(random_stencils)
average_el_var = sum(var_random_samples)/av_comp_sample_max;

println("variance at random sample in training data regime (averaged): "*string(average_el_var))

#   travelling waves
#   ================

println("\n \n travelling wave tests "*mNow()*"\n")

wave_no = 1
kappa = 2*pi*wave_no/l 
lattice_eq_rhs = 1+ dt^2/dx^2*(cos(kappa*dx)-1) - dt^2/2 
if abs(lattice_eq_rhs)>1 
    println("The discretised PDE does not admit real valued travelling wave with this wave number for this potential/dx/dt. Discriminant ") 
    println(lattice_eq_rhs) 
end 
c_tw_discrete=acos(lattice_eq_rhs)/(kappa*dt) 
amplitude_discrete = [1.,0.] 
tw_discrete(t,x) = amplitude_discrete[1]*sin(kappa*(x-c_tw_discrete*t))+amplitude_discrete[2]*cos(kappa*(x-c_tw_discrete*t)) 
U_tw_discrete = tw_discrete.(TMesh,transpose(XMesh)) 

pUTWtrue=plotU(U_tw_discrete) 

savefig(pUTWtrue,"plots/travelling_wave_true.png")

consistencyTW_true_model=(maximum(abs.(dEL(x->[Ld_ref(x)],CollectStencils(U_tw_discrete)))))
println("consistency with DEL(Ld_ref): "*string(consistencyTW_true_model)) 

consistencyTW_ml_model=maximum(abs.(dEL(x->[Ld_ml(x)],CollectStencils(U_tw_discrete))))
println("consistency with DEL(Ld_ml): "*string(consistencyTW_ml_model))

#   predict solution initialising from travelling wave
#   ––––––––––––––––––––––––––––––––––––––––––––––––––

println("\n \n predict solition initialising from true travelling wave "*mNow()*"\n")

u0Wave = Base.Fix1(tw_discrete,0.).(XMesh);
u1Wave = Base.Fix1(tw_discrete,dt).(XMesh);

U_mlWave=PDEContinueProgressBar(Ld_ml, u0Wave, u1Wave; time_steps = N);

maxTPlot = 16
pUmlWave =plotU(U_mlWave[1:maxTPlot,:])
pUrefWave = plotU(U_tw_discrete[1:maxTPlot,:]) 

savefig(pUmlWave,"plots/predicted_evolution_waveInit.png")
savefig(pUrefWave,"plots/TW1_ref.png")

plot!(pUrefWave,title="exact TW")
plot!(pUmlWave,title="ml predicted")
pWaveSummary=plot(pUrefWave,pUmlWave)

diffTW = U_mlWave[1:maxTPlot,:]-U_tw_discrete[1:maxTPlot,:]
max_err = maximum(abs.(diffTW))
l2_err  = sqrt(sum((diffTW).^2)*dx*dt)
println("max./l2 error: "*string(max_err)*" / "*string(l2_err))

function StdU(U)
    variance_Uml2 = var_el(CollectStencils(U))
    st_deviation_Uml2=sqrt.((x->max(0.,x)).(variance_Uml2)); # map numerically negative entries to 0
    st_deviation_Uml2_spatial = [zeros(2,M)
    return reshape(st_deviation_Uml2,(size(U,2),size(U,1)-2))']
end

st_deviation_UmlWave = StdU(U_mlWave);

println("max. standard deviation: "*string(maximum(st_deviation_UmlWave)))

pUml2Heat=heatU(st_deviation_UmlWave)
plot!(right_margin=40Plots.pt, colorbar=true)

pUml2HeatLog=heatU(log10.(st_deviation_UmlWave))
plot!(colorbar=true)

savefig(pUml2Heat,"plots/TW_ml_stddev_heatmap.pdf")
savefig(pUml2HeatLog,"plots/TW_ml_stddevLog_heatmap.pdf")

av_comp_sample_max=50
training_min = minimum(U_tw_discrete)
training_span=maximum(U_tw_discrete)-minimum(U_tw_discrete)
random_stencils = training_span.*rand(rng,av_comp_sample_max,7) .+ training_min;
var_random_samples=var_el(random_stencils)
average_el_var = sum(var_random_samples)/av_comp_sample_max

println("variance at random sample in regime of the TW (averaged): "*string(average_el_var))

println("end of script "*mNow())
