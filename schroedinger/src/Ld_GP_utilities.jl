function conjp_local(Ld,uStencil::Matrix{Float64})
    
    uupleft = uStencil[1,:]
    uup = uStencil[2,:]
    uupright = uStencil[3,:]
    uleft = uStencil[4,:]
    u = uStencil[5,:]
    uright = uStencil[6,:]
    udownleft = uStencil[7,:]
    udown = uStencil[8,:]
    udownright = uStencil[9,:]
    
    return -ForwardDiff.jacobian(u->Ld([u;uup;uright;uupright])+Ld([uleft;uupleft;u;uup]),u) # specific to 9pt stencil
end

function conjp_local(Ld,uStencil::Array{Float64})
         
    conjSpec=data->conjp_local(Ld,data)
    #return dropdims(mapslices(dELSpec,uStencil,dims=2),dims=2)
    return dropdims(mapslices(conjSpec,uStencil,dims=(2,3)),dims=2)
    
end



function dEL(Ld,uStencil::Matrix{Float64})

    uupleft = uStencil[1,:]
    uup = uStencil[2,:]
    uupright = uStencil[3,:]
    uleft = uStencil[4,:]
    u = uStencil[5,:]
    uright = uStencil[6,:]
    udownleft = uStencil[7,:]
    udown = uStencil[8,:]
    udownright = uStencil[9,:]
    
    # discrete Euler-Lagrange equation
    DELpre(u) = Ld([u;uup;uright;uupright])+Ld([udown;u;udownright;uright])+Ld([udownleft;uleft;udown;u])+Ld([uleft;uupleft;u;uup])
    return ForwardDiff.jacobian(DELpre, u)
    
end

function dEL(Ld,uStencil::Array{Float64})
         
    dELSpec=data->dEL(Ld,data)
    #dropdims(mapslices(dELSpec,uStencil,dims=(2,3)),dims=2)
    
    out= mapslices(dELSpec,uStencil,dims=(2,3))
    out=permutedims(out,(3,1,2))
    return out[:]
end

function StencilToQuad(baseStencil)

    u = baseStencil[5,:]
    uright = baseStencil[6,:]
    uup = baseStencil[2,:]
    uupright = baseStencil[3,:]
    return [u;uup;uright;uupright]
end

# linear observation functionals Phi in Proposition.
# evaluation of EL operator and conjp operator at data points and base
# element in dual space
function ObservationFunct(Ld,data_Stencils,baseStencil::Matrix{Float64})
    
    elL_vals = dEL(Ld,data_Stencils)
    conjpL_vals = conjp_local(Ld,baseStencil)
    base_eval = Ld(StencilToQuad(baseStencil))
    return vcat([elL_vals; conjpL_vals'; base_eval]...)

    ## code below is slightly faster but incompatible with ForwardDiff.
    #=
    noSamples = size(data_Stencils,1);
    dimU = size(data_Stencils,3)
    out = zeros((noSamples+1)*dimU  + 1)
    #out = Vector{Union{ForwardDiff.Dual,Float64}}(undef,(noSamples+1)*dimU  + 1)
    out = Vector{ForwardDiff.Dual}(undef,(noSamples+1)*dimU  + 1)
    out[1:noSamples*dimU] = dEL(Ld,data_Stencils)
    out[noSamples*dimU+1:end-1] = conjp_local(Ld,baseStencil)
    out[end:end] = Ld(StencilToQuad(baseStencil))
    return out
    =#
end





# alternative, slower implementation of theta_k (for debugging)
function theta_K(kernel,data_Stencils,baseStencil)
    
    dimU = Int(length(baseStencil)/9)
    
    Observ(fun) = ObservationFunct(fun,data_Stencils,baseStencil) 
    size_theta = size(data_Stencils,1)*dimU + dimU+1
    
    function kObs(a)
    	return Observ( b->[kernel(a,b)])
    end
    
    #return hcat([Observ(x->[kObs(x)[j]]) for j=1:size_theta]...)
    
    Theta = zeros(size_theta,size_theta) 
    @showprogress for j = 1:size_theta
    Theta[:,j]=Observ(x->[kObs(x)[j]])
    end
    return Theta
end

# faster implementation of theta_K
function theta_k(kernel,dataM,baseSt) 

    dimU = Int(length(baseSt)/9)

    basept=StencilToQuad(baseSt)
    
    elel(a,b)=dEL( a_->dEL( b_ -> [kernel(a_,b_)],b)[:],a)
    
    function elel(Stencils::Array{Float64})
		N=size(Stencils,1)

		dimU=size(Stencils,3)
		
		ELEL = zeros(dimU*N,dimU*N) 

		@showprogress for j=0:N-1
		    for i=0:N-1
		        st1 = Stencils[j+1,:,:]
		        st2 = Stencils[i+1,:,:]
		        ELEL[j*dimU+1:j*dimU+dimU,i*dimU+1:i*dimU+dimU] = elel(st2,st1)
		    end
		end
		return ELEL
    end

    elp(a::Matrix,b::Matrix) = dEL(a_->conjp_local(b_ -> [kernel(a_,b_)],b)[:],a)
    function elp(Stencils::Array,baseStencil)
	    N= size(Stencils,1)
	    return hcat([elp(Stencils[j,:,:],baseStencil) for j=1:N]...)
	end
	
	pel(a,b)=elp(a,b)'
    #pel(a,b) = conjp_local(a_->dEL( b_ -> [kernel(a_,b_)],b),a)
    pp(a,b) = conjp_local(a_->conjp_local( b_ -> [kernel(a_,b_)],b)[:],a)
    
    elev(a,b) = dEL(a_->[kernel(a_,b)],a)
    pev(a,b) = conjp_local(a_->[kernel(a_,b)],a)
    
    # evaluations
    elelM = elel(dataM)
    elpM = elp(dataM,baseSt)
    #pelM = pel(baseSt,dataM)
    pelM = elpM'
    ppM = pp(baseSt,baseSt)
    
    elevM = elev(dataM,basept)
    pevM = pev(baseSt,basept)
    evevM = kernel(basept,basept)
    
    theta = [elelM elpM' elevM; elpM ppM pevM'; elevM' pevM evevM]
    
end


function kappaPhi(kernel,data_Stencils,baseStencil)

    function ObsKernelA(a)
        k1a(b) = [kernel(a,b)]
    return ObservationFunct(k1a,data_Stencils,baseStencil)
    end

    return ObsKernelA
end


# Posterior computation GP with linear constraints
# See textbook by Owhadi, Scovel 2019 https://dx.doi.org/10.1017/9781108594967
function Lagrangian_Gamblet(kernel,data_Stencils,baseStencil;normalisation_balancing=ones(2))
    
    Theta = theta_k(kernel,data_Stencils,baseStencil)
    Theta_fact=factorize(Theta)
    
    
    
    # Lagrangian as conditional mean
    #KappaPhi = kappaPhi(kernel,data_Stencils,baseStencil) # Element in RKHS. This is a function handle.
    #function Ld_ml(ujet)
    #    dimU = Int(length(ujet)/4)
    #    gamblets_values = Theta_fact\KappaPhi(ujet)
    #    return normalisation_balancing'*gamblets_values[end-dimU:end]
    #end

    

    # Lagrangian as conditional mean
    #FASTER implementation
    
    noSamples = size(data_Stencils,1);
    dimU = size(data_Stencils,3)

    function Ld_ml(ujet)
        
        out = zeros(typeof(ujet[1]),(noSamples+1)*dimU  + 1)
        ka = b->[kernel(ujet,b)]
        #out = Vector{Union{ForwardDiff.Dual,Float64}}(undef,(noSamples+1)*dimU  + 1)
        #out = Vector{ForwardDiff.Dual}(undef,(noSamples+1)*dimU  + 1)
        out[1:noSamples*dimU] = dEL(ka,data_Stencils)
        out[noSamples*dimU+1:end-1] = conjp_local(ka,baseStencil)
        out[end:end] = ka(StencilToQuad(baseStencil))
        gamblets_values = Theta_fact\out
        return normalisation_balancing'*gamblets_values[end-dimU:end]
    end


    
    function Ld_ml(u,uup,uright,uupright)
        return Ld_ml([u; uup; uright; uupright])
    end
    
    function dEL_ml(uStencil)
        return dEL(x->[Ld_ml(x)],uStencil)
    end
    
    
    # CONDITIONAL VARIANCE
    # covariance operators for various observables
    
    # variance of observable psi in U^ast
	function varU(psi)

		kappa_phi = a->psi(b->kernel(a,b))
		kappa_phi_phi = psi(kappa_phi)
		kappa_phi_Phi = ObservationFunct(a->[kappa_phi(a)],data_Stencils,baseStencil)
		
		linSys = Theta_fact\kappa_phi_Phi
		
		return kappa_phi_phi .- kappa_phi_Phi'*linSys
		
	end
	
	# returns function varf : domain(fun) -> reals, varf(z) = varU(fun(z))
	function varFun(fun)
		function varf(z)
		    psi = fun(z) # get element in RKHS
		return varU(psi)
		end
		return varf
	end 
        
    # variance point evaluation
    variance_pts=varFun(x->(L->L(x)))
    
    # variance first DEL equation
	#variance_dEL1 = varFun(stencil -> (Ld-> dEL(x->[Ld(x)],stencil)[1]))
    
    # variance DEL
    variance_DEL(stencil) = [varU(Ld-> dEL(x->[Ld(x)],stencil)[j]) for j=1:dimU]
    
    return Ld_ml, dEL_ml, varFun, variance_pts, variance_DEL, Theta, Theta_fact

end

