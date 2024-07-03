"""
    DELSolver(Ld_ml)

Defines 9 point stencil and method of line - type solver functions to discrete Lagrangian Ld_ml
"""
function DELSolver9ptStencil(Ld_ml)



    
    function Ld_ml_d1(a,b,c,d)
    return  ForwardDiff.gradient(a->Ld_ml(a,b,c,d),a)
    end

    function Ld_ml_d2(a,b,c,d)
        return  ForwardDiff.gradient(b->Ld_ml(a,b,c,d),b)
    end

    function Ld_ml_d3(a,b,c,d)
        return  ForwardDiff.gradient(c->Ld_ml(a,b,c,d),c)
    end

    function Ld_ml_d4(a,b,c,d)
        return  ForwardDiff.gradient(d->Ld_ml(a,b,c,d),d)
    end


    function DELdx_ml_stencil(uupleft,uup,uupright, uleft,u,uright, udownleft,udown,udownright)
        return Ld_ml_d1(u,uup,uright,uupright)+Ld_ml_d2(udown,u,udownright,uright)+Ld_ml_d4(udownleft,uleft,udown,u)+Ld_ml_d3(uleft,uupleft,u,uup)
    end

    function DELdx_ml_stencil(stencil)
        uupleft =  stencil[:,1]
        uup  =   stencil[:,2]
        uupright  =   stencil[:,3]
        uleft  =  stencil[:,4]
        u  =  stencil[:,5]
        uright =  stencil[:,6]
        udownleft  =  stencil[:,7]
        udown  =  stencil[:,8]
        udownright  =  stencil[:,9]
        return DELdx_ml_stencil(uupleft,uup,uupright, uleft,u,uright, udownleft,udown,udownright)
    end

    function DELdx_ml(U0,U1,U2)

        M = size(U0,2)
        dimU = size(U0,1)
        modInd(i::Int) = mod(i-1,M)+1

        #out = zeros(dimU*M)
        out = similar(U2)

        for j=1:M
        #for j in axes(U0,2)

            uupleft =  U2[:,modInd(j-1)]
            uup  =  U2[:,modInd(j)]
            uupright  =  U2[:,modInd(j+1)]
            uleft  =  U1[:,modInd(j-1)]
            u  =  U1[:,modInd(j)]
            uright =  U1[:,modInd(j+1)]
            udownleft  =  U0[:,modInd(j-1)]
            udown  =  U0[:,modInd(j)]
            udownright  =  U0[:,modInd(j+1)]

            #out[1+dimU*(j-1):dimU*(j-1)+dimU] = DELdx_ml_stencil(uupleft,uup,uupright, uleft,u,uright, udownleft,udown,udownright)
            out[:,j] = DELdx_ml_stencil(uupleft,uup,uupright, uleft,u,uright, udownleft,udown,udownright)

        end
        
        return out

    end

    function DELdx_ml!(out,U0,U1,U2)

        M = size(U0,2)
        #dimU = size(U0,1)
        modInd(i::Int) = mod(i-1,M)+1

        #out = similar(U2)

        for j=1:M

            uupleft =  U2[:,modInd(j-1)]
            uup  =  U2[:,modInd(j)]
            uupright  =  U2[:,modInd(j+1)]
            uleft  =  U1[:,modInd(j-1)]
            u  =  U1[:,modInd(j)]
            uright =  U1[:,modInd(j+1)]
            udownleft  =  U0[:,modInd(j-1)]
            udown  =  U0[:,modInd(j)]
            udownright  =  U0[:,modInd(j+1)]

            out[:,j] = DELdx_ml_stencil(uupleft,uup,uupright, uleft,u,uright, udownleft,udown,udownright)

        end
        
        #return out

    end

    function DELObjectiveDiff!(Out,U0,U1,U2)
        Out[:] = ForwardDiff.jacobian(U2->DELdx_ml(U0,U1,U2),U2)
    end


    function DELSolve(U0,U1)
        guess = 2*U1-U0
        DELObjectiveInstance!(Out,U2) = DELdx_ml!(Out,U0,U1,U2)
        DELObjectiveDiffInstance!(Out,U2) = DELObjectiveDiff!(Out,U0,U1,U2)
        return nlsolve(DELObjectiveInstance!,DELObjectiveDiffInstance!,guess)
    end

    function DELSolve(q0,q1,steps)

        trj = zeros((size(q0)...,steps+1))
        trj[:,:,1]=q0
        trj[:,:,2]=q1
        
        @showprogress for j = 1:steps-1        
            trj[:,:,j+2] = DELSolve(trj[:,:,j],trj[:,:,j+1]).zero
        end
        
        return trj
    end

    return DELSolve, DELdx_ml

end