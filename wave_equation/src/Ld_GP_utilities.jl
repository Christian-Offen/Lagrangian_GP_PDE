function conjp_local(Ld,uStencil::Vector{Float64})
    
    u,uup,uleft,udown,uright,uupleft,udownright = uStencil
    
    #Ld_ref(u,udown,uleft)
    return jacobian(b->Ld([u; b; uleft]),[udown]) # specific to 7pt stencil that this is so easy
end

function conjp_local(Ld,uStencil::Matrix{Float64})
         
    conjSpec=data->conjp_local(Ld,data)
    #return dropdims(mapslices(dELSpec,uStencil,dims=2),dims=2)
    return mapslices(conjSpec,uStencil,dims=2)
    
end



function dEL(Ld,uStencil::Vector{Float64})
         
    u,uup,uleft,udown,uright,uupleft,udownright = uStencil
    
    # discrete Euler-Lagrange equation
    DELpre(u) = Ld([u; udown; uleft])+Ld([uup; u; uupleft])+Ld([uright; udownright;u])
    return jacobian(DELpre, [u])
    
end

function dEL(Ld,uStencil::Matrix{Float64})
         
    dELSpec=data->dEL(Ld,data)
    #return dropdims(mapslices(dELSpec,uStencil,dims=2),dims=2)
    return mapslices(dELSpec,uStencil,dims=2)
    
end



# linear observation functionals Phi in Proposition.
# evaluation of EL operator and conjp operator at data points and base
# element in dual space
function ObservationFunct(Ld,data_Stencils,baseStencil)
    
    elL_vals = dEL(Ld,data_Stencils)
    conjpL_vals = conjp_local(Ld,baseStencil);
    base_eval = Ld(baseStencil[1:3])
    
    return vcat([elL_vals; conjpL_vals; base_eval]...)
end


# alternative, slower implementation of theta_k(kernel,data_TQ2,basept)
function theta_K(kernel,data_Stencils,baseStencil)
    
    dimU = Int(length(baseStencil)/7)
    
    Observ(fun) = ObservationFunct(fun,data_Stencils,baseStencil) 
    size_theta = size(data_Stencils,1) + dimU+1
    
    kObs= a-> Observ( b->[kernel(a,b)])
    
    return hcat([Observ(x->[kObs(x)[j]]) for j=1:size_theta]...)
    
end

# faster implementation of theta_K
function theta_k(kernel,dataM,baseSt) 

    dimU = Int(length(baseSt)/7)
    basept = baseSt[1:3*dimU]
    
    elel(a,b)= dEL(a_->dEL( b_ -> [kernel(a_,b_)],b),a)
    elp(a,b) = dEL(a_->conjp_local(b_ -> [kernel(a_,b_)],b),a)
    pel(a,b) = conjp_local(a_->dEL( b_ -> [kernel(a_,b_)],b),a)
    pp(a,b) = conjp_local(a_->conjp_local( b_ -> [kernel(a_,b_)],b),a)
    
    elev(a,b) = dEL(a_->[kernel(a_,b)],a)
    pev(a,b) = conjp_local(a_->[kernel(a_,b)],a)
    
    # evaluations
    elelM = elel(dataM,dataM)
    elpM = elp(dataM,baseSt)
    pelM = pel(baseSt,dataM)
    ppM = pp(baseSt,baseSt)
    
    elevM = elev(dataM,basept)
    pevM = pev(baseSt,basept)
    evevM = kernel(basept,basept)
    
    theta = [elelM vcat(elpM...) vcat(elevM...); hcat(pelM...) ppM pevM; vcat(elevM...)' pevM' evevM]
end


function kappaPhi(kernel,data_Stencils,baseStencil)

    function ObsKernelA(a)
        k1a(b) = [kernel(a,b)]
    return ObservationFunct(k1a,data_Stencils,baseStencil)
    end

    return ObsKernelA
end





# Justified by Prop2.1 Chen, Hosseini, Owhadi, Stuart: Solving and learning nonlinear PDEs with Gaussian Processes
function Lagrangian_Posterior(kernel,data_Stencils,baseStencil;normalisation_balancing=ones(2))
    
    Theta = theta_k(kernel,data_Stencils,baseStencil)
    Theta_fact=factorize(Theta)
    
    KappaPhi = kappaPhi(kernel,data_Stencils,baseStencil) # Element in RKHS. This is a function handle.
    
    
    # Lagrangian as conditional mean
    function Ld_ml(ujet)
        dimU = Int(length(ujet)/3)
        gamblets_values = Theta_fact\KappaPhi(ujet)
        return normalisation_balancing'*gamblets_values[end-dimU:end]
    end
    
    function Ld_ml(u,udown,uleft)
        return Ld_ml([u; udown; uleft])
    end
    
    function dEL_ml(uStencil)
        return dEL(x->[Ld_ml(x)],uStencil)
    end
    
    
    # CONDITIONAL VARIANCE
    # covariance operators for various observables
    
    # variance for observation variable operator; operator(#1)(qjet) \in U^*     (U^* = dual of RKHS)
    function var_ml_operator(operator)
        
        function var_ml(ujet)
        
            operator_handle(b) = operator(a->[kernel(a,b)])(ujet)
            kappa_Phi_phi = ObservationFunct(operator_handle,data_Stencils,baseStencil)
            linSys = Theta_fact\kappa_Phi_phi

            kappa_phi_phi = operator(operator_handle)(ujet)

            return kappa_phi_phi .- kappa_Phi_phi'*linSys
        end
        
        function var_ml(u,udown,uleft)
            return var_ml([u; udown; uleft])
        end
        
        function var_ml(ujets::Matrix{Float64})
            return mapslices(var_ml,ujets,dims=2)
        end
        
        return var_ml
        
    end
    
    # point evaluation
    var_ml_pts = var_ml_operator(L-> (x ->L(x)))        
    
    # conjugate momenta
    var_ml_p=var_ml_operator(L-> (x -> conjp_local(L,x)))
    
    # DEL
    var_ml_el=var_ml_operator(Ld->(x->dEL(Ld,x)))
        
    return Ld_ml, dEL_ml, var_ml_operator, var_ml_pts, var_ml_p, var_ml_el, Theta, Theta_fact

end

