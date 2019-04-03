using LinearAlgebra
using Parameters
using IterativeSolvers
using FastGaussQuadrature
using ForwardDiff
using QuantEcon
using Plots
using Arpack
include("support.jl")
################################### Model types #########################
###I can
Ud(c,γ) = c^(-γ)
Udd(c,γ) = -γ*c^(-γ-1.0)
uPrimeInv(up,γ) = up.^(-1.0/γ)

mutable struct AiyagariParameters{T <: Real}
    β::T
    α::T
    δ::T
    γ::T
    ρ::T
    σz::T #st. deviation of Z shock
    σ::T #job separation
    lamw::T #job finding prob
    Lbar::T
    amin::T
    Penalty::T
end

struct FiniteElement{R <: Real,I <: Integer}
    m::I
    wx::Array{R,1}
    ax::Array{R,1}
end

mutable struct AiyagariModel{T <: Real,I <: Integer}
    params::AiyagariParameters{T}
    aGrid::AbstractArray ##Policy grid
    na::I ##number of grid points in policy function
    dGrid::AbstractArray
    nd::I ##number of grid points in distribution
    states::AbstractArray ##earning states 
    ns::I ##number of states
    EmpTrans::AbstractArray
end
mutable struct AggVars{T <: Real}
    R::T
    w::T
end

function Prices(K,Z,params::AiyagariParameters)
    @unpack β,α,δ,γ,ρ,σ,lamw,Lbar = params
    R = Z*α*(K/Lbar)^(α-1.0) + 1.0 - δ
    w = Z*(1.0-α)*(K/Lbar)^(1.0-α) 
    
    return AggVars(R,w)
end

function Aiyagari(
    K::T,
    β::T = 0.98,
    α::T = 0.4,
    δ::T = 0.02,
    γ::T = 2.0,
    ρ::T = 0.95,
    σz::T = 1.0,
    σ::T = 0.2,
    lamw::T = 0.6,
    Lbar::T = 1.0,
    amin::T = 0.0,
    amax::T = 200.0,
    Penalty::T = 10000000000.0,
    na::I = 201,
    nd::I = 201,
    ns::I = 2,
    endow = [1.0;2.5],
    NumberOfQuadratureNodesPerElement = 2) where{T <: Real,I <: Integer}

    #############Params
    params = AiyagariParameters(β,α,δ,γ,ρ,σz,σ,lamw,Lbar,amin,Penalty)
    AggVars = Prices(K,1.0,params)
    @unpack R,w = AggVars

    ################## Policy grid
    function grid_fun(a_min,a_max,na, pexp)
        x = range(a_min,step=0.5,length=na)
        grid = a_min .+ (a_max-a_min)*(x.^pexp/maximum(x.^pexp))
        return grid
    end
    aGrid = grid_fun(amin,amax,na,4.0)
    ################### Distribution grid
    dGrid=aGrid

    ################## Transition
    EmpTrans = [1.0-lamw σ ;lamw 1.0-σ]
    dis = LinearAlgebra.eigen(EmpTrans)
    mini = argmin(abs.(dis.values .- 1.0)) 
    stdist = abs.(dis.vectors[:,mini]) / sum(abs.(dis.vectors[:,mini]))
    lbar = dot(stdist,endow)
    states = endow/lbar

    @assert sum(EmpTrans[:,1]) == 1.0 ###sum to 1 across rows
    #summing across rows is nice as we don't need to transpose transition before taking eigenvalue
    
    ################### Final model pieces
    Guess = zeros(na*ns)
    for (si,earn) in enumerate(states)
        for (ki,k) in enumerate(aGrid) 
            n = (si-1)*na + ki
            ap = R*β*k + earn - 0.001*k
            Guess[n] = ap
        end
    end
    GuessMatrix = reshape(Guess,na,ns)

    QuadratureAbscissas,QuadratureWeights = gausslegendre(NumberOfQuadratureNodesPerElement)
    finel = FiniteElement(NumberOfQuadratureNodesPerElement,QuadratureWeights,QuadratureAbscissas)
    
    return AiyagariModel(params,aGrid,na,dGrid,nd,states,ns,EmpTrans),Guess,AggVars,finel
end

function Residual(
    pol::AbstractArray,
    AggVars::AggVars,
    aiyagari::AiyagariModel,
    Penalty)
    
    #Model parameters
    @unpack params,states,ns,na,aGrid,EmpTrans = aiyagari
    @unpack β,γ,Penalty = params
    R,w = AggVars.R,AggVars.w

    #na = size(Grid,1)

    #nx = na*ns
    Resid = zeros(eltype(pol),ns*na)
    #dResid = zeros(eltype(pol),ns*na,ns*na)
    #r = Float64[]
    #ri = Integer[]
    rd = Float64[]
    rdcoli = Int64[]
    rdrowi = Int64[]
    #fnsearch = zeros(Int64,2*ns)
    np = 0
    fn = 0
    for (s,earn) in enumerate(states)
        for (n,a) in enumerate(aGrid)
            s1 = (s-1)*na + n
            #push!(ri,s1)
            #Policy functions
            ap = pol[s1]
            #push!(r,ap)
            #penalty function
            pen = Penalty*min(ap,0.0)^2
            dpen = 2*Penalty*min(ap,0.0)
            c = R*a + earn*w - ap

            #preferences
            uc = Ud(c,γ)
            ucc = Udd(c,γ)
            ∂c∂ai = -1.0
            
            basisp1,basisp2,np,dbasisp1 = lininterp(aGrid,ap,na)
            dbasisp2 = -dbasisp1                
            #@show n
            tsai = 0.0
            sum1 = 0.0
            #sp1,sp2 = 0,0
            #fnsearch = Int64[]
            for (sp,earnp) in enumerate(states)
                sp1 = (sp-1)*na + np
                sp2 = (sp-1)*na + np + 1
                
                #Policy functions
                app = pol[sp1]*basisp1 + pol[sp2]*basisp2
                cp = R*ap + earnp*w - app
                ucp = Ud(cp,γ)
                uccp = Udd(cp,γ)

                #Need ∂cp∂ai and ∂cp∂aj
                ∂ap∂ai = pol[sp1]*dbasisp1 + pol[sp2]*dbasisp2
                ∂cp∂ai = R - ∂ap∂ai
                ∂cp∂aj = -1.0

                sum1 += β*(EmpTrans[sp,s]*R*ucp + pen)

                #summing derivatives with respect to θs_i associated with c(s)
                tsai += β*(EmpTrans[sp,s]*R*uccp*∂cp∂ai + dpen)
                tsaj = β*EmpTrans[sp,s]*R*uccp*∂cp∂aj
                #dResid[s1,sp1] += tsaj * basisp1
                #dResid[s1,sp2] += tsaj * basisp2

                
                push!(rdcoli,sp1)
                push!(rdrowi,s1)
                push!(rd,tsaj * basisp1)
                #push!(fnsearch = sp1

                
                push!(rdcoli,sp2)
                push!(rdrowi,s1)
                push!(rd,tsaj * basisp2)
                
            end
            ##add the LHS and RHS of euler for each s wrt to θi
            dres = tsai - ucc*∂c∂ai

            #for each row, the columns accessed through sp1 and sp2 are unique. fnsearch is the vector with those ns*2 indices, then we just have to see if s1 coincides with one of the 2*ns indices, if, so just add its value to it and don't add another col,row index  
            fnsearch = rdcoli[end-(2*ns-1):end]
            for i = 1:2*ns
                fnsearch[i] == s1 ? fn = i : fn = 0
            end

            if fn == 0
                push!(rdcoli,s1)
                push!(rdrowi,s1)
                push!(rd,dres)
            else
                rd[end-(2*ns-fn)] += dres
            end
 
            res = sum1 - uc
            Resid[s1] = res
        end
    end 
    Resid,SparseArrays.sparse(rdrowi,rdcoli,rd)
end

function WeightedResidual(
    pol,
    AggVars::AggVars,
    aiyagari::AiyagariModel,
    Penalty,
    FiniteElement::FiniteElement)

    @unpack params,aGrid,na,dGrid,nd,states,ns,EmpTrans = aiyagari
    @unpack β,α,δ,γ,ρ,σ,lamw,Lbar,amin,Penalty = params
    R,w = AggVars.R,AggVars.w
    nx = na*ns
    @unpack m,wx,ax = FiniteElement 
    Resid = zeros(eltype(pol),ns*na)
    dResid = zeros(eltype(pol),ns*na,ns*na)
    
    ne = na-1
    np = 0    
    for (s,earn) in enumerate(states)
        for n=1:ne
            a1,a2 = aGrid[n],aGrid[n+1]
            s1 = (s-1)*na + n
            s2 = (s-1)*na + n + 1
            for i=1:m
                #transforming k according to Legendre's rule
                a = (a1 + a2)/2.0 + (a2 - a1)/2.0 * ax[i]
                v = (a2-a1)/2.0*wx[i]

                #Form basis for piecewise function
                basis1 = (a2 - a)/(a2 - a1)
                basis2 = (a - a1)/(a2 - a1)

                #Policy functions
                ap = pol[s1]*basis1 + pol[s2]*basis2
                c = R*a + earn*w - ap

                #penalty function
                uc = Ud(c,γ)
                ucc = Udd(c,γ)                    
                ∂c∂ai = -1.0
                pen = Penalty*min(ap,0.0)^2
                dpen = 2*Penalty*min(ap,0.0)

                basisp1,basisp2,np,dbasisp1 = lininterp(aGrid,ap,na)
                dbasisp2 = -dbasisp1                
                
                tsai = 0.0
                sum1 = 0.0
                for (sp,earnp) in enumerate(states)
                    sp1 = (sp-1)*na + np
                    sp2 = (sp-1)*na + np + 1

                    #Policy functions
                    app = pol[sp1]*basisp1 + pol[sp2]*basisp2
                    cp = R*ap + earnp*w - app

                    #agents
                    ucp = Ud(cp,γ)
                    uccp = Udd(cp,γ)                    
 
                    
                    #Need ∂cp∂ai and ∂cp∂aj
                    ∂ap∂a = pol[sp1]*dbasisp1 + pol[sp2]*dbasisp2
                    ∂cp∂ai = R - ∂ap∂a 
                    ∂cp∂aj = -1.0

                    sum1 += β*(EmpTrans[sp,s]*R*ucp + pen) 
                    
                    tsai += β*(EmpTrans[sp,s]*R*uccp*∂cp∂ai + dpen)
                    tsaj = β*EmpTrans[sp,s]*R*uccp*∂cp∂aj

                    dResid[s1,sp1] +=  basis1 * v * tsaj * basisp1
                    dResid[s1,sp2] +=  basis1 * v * tsaj * basisp2
                    dResid[s2,sp1] +=  basis2 * v * tsaj * basisp1
                    dResid[s2,sp2] +=  basis2 * v * tsaj * basisp2 
                end
                ##add the LHS and RHS of euler for each s wrt to θi
                dres =  tsai - ucc*∂c∂ai
                
                dResid[s1,s1] +=  basis1 * v * dres * basis1
                dResid[s1,s2] +=  basis1 * v * dres * basis2
                dResid[s2,s1] +=  basis2 * v * dres * basis1
                dResid[s2,s2] +=  basis2 * v * dres * basis2

                res = sum1 - uc


                Resid[s1] += basis1*v*res
                Resid[s2] += basis2*v*res
            end
        end
    end 
    Resid,dResid 
end


function SolveCollocation(
    pol::AbstractArray,
    AggVars::AggVars{T},
    aiyagari::AiyagariModel,
    Penalty,
    maxn::Int64 = 150,
    tol = 1e-12
) where{T <: Real,I <: Integer}

    @unpack ns,na = aiyagari
    kink = Int64[]


    @unpack ns,na = aiyagari 
    for i = 1:maxn
        Res,dRes = Residual(pol,AggVars,aiyagari,Penalty)
        step = - dRes \ Res
        #step = -gmres!(zeros(pol), dRes, Res; maxiter=10000)
        
        if LinearAlgebra.norm(step) > 1.0
            pol += 1.0/2.0*step
        else
            pol += 1.0/1.0*step
        end
        #@show maximum(abs.(step))
        if maximum(abs.(step)) < tol
            println("Individual problem converged in ",i," steps")
            return pol
            break
        end
    end
    return println("Individual problem Did not converge")
end
function SolveFiniteElement(
    pol,
    AggVars,
    aiyagari,
    Penalty,
    FiniteElement::FiniteElement,
    maxn::Int64 = 100,
    tol = 1e-12
) where{T <: Real,I <: Integer}

    @unpack ns,na = aiyagari
    kink = Int64[]

    for i = 1:maxn
        Res,dRes = WeightedResidual(pol,AggVars,aiyagari,Penalty,FiniteElement)
        #testing
        #dRes = ForwardDiff.jacobian(t -> WeightedResidual(t,AggVars,aiyagari,Penalty,Res,dRes,FiniteElement)[1],pol)
        step = - dRes \ Res
        if LinearAlgebra.norm(step) > 1.0
            pol += 1.0/2.0*step
        else
            pol += 1.0/1.0*step
        end
        @show maximum(abs.(step))
        if maximum(abs.(step)) < tol
            return pol
            break
        end
    end

    return println("Individual problem Did not converge")
end



function EulerResErr(
    Agg::AggVars,
    pol::AbstractArray,
    pol_P::AbstractArray,
    aiyagari::AiyagariModel,
    EE::AbstractArray,
    Grid::AbstractArray)

    @unpack params,aGrid,na,dGrid,nd,states,ns,EmpTrans = aiyagari
    @unpack β,α,δ,γ,ρ,σ,lamw,Lbar,amin,Penalty = params

    R,w = Agg.R,Agg.w
    #R_P,w_P = Agg_P.R, Agg_P.w
    apol = fill(0.0,ns*length(Grid))
    ##########################################################
    # For the pieces from FEM
    ##########################################################
    for (s,earn) in enumerate(states)
        for (i,a) in enumerate(Grid)

            #find basis for policy
            basis1,basis2,n = lininterp(aGrid,a,na,false)
            s1 = (s-1)*na + n
            s2 = (s-1)*na + n+1
            apol[(s-1)*length(Grid)+i] = basis1*pol[s1] + basis2*pol[s2]
            ap = apol[(s-1)*length(Grid)+i]
            
            pen = Penalty*min(ap,0.0)^2
            c = R*a + w*earn - ap
            uc = Ud(c,γ)

            #New policy basis functions
            basisp1,basisp2,np = lininterp(aGrid,ap,na,false)            
            
            ee_rhs = 0.0
            for (sp,earnp) in enumerate(states)
                sp1 = (sp-1)*na + np
                sp2 = (sp-1)*na + np + 1
                
                #Policy
                app = pol_P[sp1]*basisp1 + pol_P[sp2]*basisp2
                cp = R*ap + w*earnp - app
                ucp = cp^(-γ)

                ###Euler RHS
                ee_rhs += β*(EmpTrans[sp,s]*R*ucp + pen)  
            end

            res = uc - ee_rhs
            EE[(s-1)*length(Grid)+i] = res
        end
    end 

    return EE,apol
end



function lininterp(x,x1,sizex,derivs=true)
    n = searchsortedlast(x,x1)

    #extend linear interpolation and assign edge indices
    (n > 0 && n < sizex) ? nothing : 
        (n == sizex) ? n = sizex-1 : 
             n = 1 

    xl,xh = x[n],x[n+1]
    basis1 = (xh - x1)/(xh - xl)
    basis2 = (x1 - xl)/(xh - xl)

    if derivs
        dbasis1 =  -1.0/(xh-xl) 
        return basis1,basis2,n,dbasis1
    else
        return basis1,basis2,n
    end
end
     

function TransMat(pol,aiyagari::AiyagariModel,AggVars::AggVars,tmat) 
    
    @unpack params,aGrid,na,dGrid,nd,states,ns,EmpTrans = aiyagari
    @unpack R,w = AggVars
    
    nf = ns*nd
    pol = reshape(pol,na,ns)

    for s=1:ns
        for (i,x) in enumerate(dGrid)            
            ######
            # find each k in dist grid in nodes to use FEM solution
            ######
            n = searchsortedlast(aGrid,x)
            (n > 0 && n < na) ? n = n : 
                (n == na) ? n = na-1 : 
                    n = 1 
            x1,x2 = aGrid[n],aGrid[n+1]
            basis1 = (x2 - x)/(x2 - x1)
            basis2 = (x - x1)/(x2 - x1)
            ap  = basis1*pol[n,s] + basis2*pol[n+1,s]            
            
            ######
            # Find in dist grid where policy function is
            ######            
            np = searchsortedlast(dGrid,ap)
            aph_id,apl_id = np + 1, np
            if np > 0 && np < nd
                aph,apl = dGrid[np+1],dGrid[np]
                ω = (ap - apl)/(aph - apl)
            end
            
            
            ######            
            for si = 1:ns
                aa = (s-1)*nd + i
                ss = (si-1)*nd + np
                if np > 0 && np < nd                    
                    tmat[ss+1,aa] = EmpTrans[si,s]*ω
                    tmat[ss,aa]  = EmpTrans[si,s]*(1.0 - ω)
                elseif np == 0
                    ω = 1.0
                    tmat[ss+1,aa] = EmpTrans[si,s]*ω
                else
                    ω = 1.0
                    tmat[ss,aa] = EmpTrans[si,s]*ω
                end
            end
        end
    end
    
    tmat
end

function equilibrium(aiyagari::AiyagariModel,K0,pol0,tmat,FiniteElement,tol=1e-10,maxn=100)

    @unpack params,aGrid,dGrid,na,ns,nd,states = aiyagari

    #AssetDistribution = zeros(nd*ns)
    EA = 0.0
    
    ###Start Bisection
    uK,lK = K0, 0.0

    print("Iterate on aggregate assets")
    for kit = 1:maxn
        Aggs = Prices(K0,1.0,params)

        #solve individual problem
        pol0 = SolveCollocation(pol0,Aggs,aiyagari,params.Penalty)

        #Get transition matrix
        tmat .= 0.0
        Qa = TransMat(pol0,aiyagari,Aggs,tmat)

        #stationary distribution
        λ, x = powm!(Qa, rand(ns*nd), maxiter = 10000,tol = 1e-9)
        #@show λ
        x = x/sum(x)
        EA = dot(vcat(dGrid,dGrid),x)

        #begin iteration
        if (EA > K0) ### too little lending -> low r -> too much borrowing 
            uK = min(EA,uK)  
            lK = max(K0,lK)
            K0 = 1.0/2.0*(lK + uK)
        else ## too much lending -> high r -> too little borrowing
            uK = min(K0,uK)
            lK = max(EA,lK)
            K0 = 1.0/2.0*(lK + uK)
        end
        #@show eltype(K0)
        #println("Interest rate: ",Aggs.R," ","Bonds: ",EA)
        if abs(EA - K0) < 1e-7
            println("Markets clear!")
            println("Interest rate: ",Aggs.R," ","Bonds: ",EA)
            polm = reshape(pol0,na,ns)
            cpol = Aggs.R*hcat(aGrid,aGrid) .+ Aggs.w*repeat(reshape(states,(1,ns)),outer=[na,1]) - polm  
            return pol0,polm,EA,Aggs.R,x,reshape(x,nd,ns),cpol,Aggs
            break
        end
    end
    
    return println("Markets did not clear")
end

function F(X_L::AbstractArray,
           X::AbstractArray,
           X_P::AbstractArray,
           epsilon::AbstractArray,
           aiyagari::AiyagariModel,pos)

    @unpack params,na,nd,ns = aiyagari
    
    m = na*ns
    md = nd*ns
    pol_L,dist_L,Agg_L = X_L[1:m],X_L[m+1:m+md-1],X_L[m+md:end]
    pol,dist,Agg = X[1:m],X[m+1:m+md-1],X[m+md:end]
    pol_P,dist_P,Agg_P = X_P[1:m],X_P[m+1:m+md-1],X_P[m+md:end]

    K_L, Z_L = Agg_L
    K, Z = Agg
    K_P, Z_P = Agg_P
    
    D_L = vcat(1.0-sum(dist_L),dist_L)
    D   = vcat(1.0-sum(dist),dist)
    D_P = vcat(1.0-sum(dist_P),dist_P)
    
    Price = Prices(K_L,Z,params)
    Price_P = Prices(K,Z_P,params)
    
    #Need matrices that pass through intermediate functions to have the same type as the
    #argument of the derivative that will be a dual number when using forward diff. In other words,
    #when taking derivative with respect to X_P, EE, his, his_rhs must have the same type as X_P
    if pos == 1 
        EE = zeros(eltype(X_L),ns*na)
        his = zeros(eltype(X_L),ns*nd)
        tmat = zeros(eltype(X_L),(ns*na,ns*na))
    elseif pos == 2
        EE = zeros(eltype(X),ns*na)
        his = zeros(eltype(X),ns*nd)
        tmat = zeros(eltype(X),(ns*na,ns*na))
    else
        EE = zeros(eltype(X_P),ns*na)
        his = zeros(eltype(X_P),ns*nd)
        tmat = zeros(eltype(X_P),(ns*na,ns*na))
    end
    agg_root = AggResidual(D,K,Z_L,Z,epsilon,aiyagari)
    dist_root = WealthResidual(pol_L,D_L,D,aiyagari,tmat,his) ###Price issue
    euler_root = EulerResidual(Price,Price_P,pol,pol_P,aiyagari,EE)
    
    return vcat(euler_root,dist_root,agg_root)
end

function AggResidual(D::AbstractArray,K,Z_L,Z,epsilon::AbstractArray,aiyagari::AiyagariModel)
    @unpack params,dGrid = aiyagari
    @unpack ρ,σz = params
    ϵz = epsilon[1]

    AggAssets = dot(D,vcat(dGrid,dGrid))
    AggEqs = vcat(
        AggAssets - K, #bond market clearing
        log(Z) - ρ*log(Z_L) - σz*ϵz, #TFP evol
    ) 
    
    return AggEqs
end

function WealthResidual(pol::AbstractArray,
                        Dist_L::AbstractArray,
                        Dist::AbstractArray,
                        aiyagari::AiyagariModel,
                        tmat::AbstractArray,
                        his::AbstractArray)
    
    Qa = TransMat(pol,aiyagari,Aggs,tmat)
    for i in eachindex(his)
        his[i] = Dist[i] - dot(Qa[i,:],Dist_L)
    end
    return his[2:end]
end

function EulerResidual(
    Agg::AggVars,
    Agg_P::AggVars,
    pol::AbstractArray,
    pol_P::AbstractArray,
    aiyagari::AiyagariModel,
    EE::AbstractArray)


    @unpack params,aGrid,na,dGrid,nd,states,ns,EmpTrans = aiyagari
    @unpack β,α,δ,γ,ρ,σ,lamw,Lbar,amin,Penalty = params

    R,w = Agg.R,Agg.w
    R_P,w_P = Agg_P.R, Agg_P.w
    
    ##########################################################
    # For the pieces from FEM
    ##########################################################
    for (s,earn) in enumerate(states)
        for i=1:na
            s1 = (s-1)*na + i
            a = aGrid[i]
            
            #Policy functions
            ap = pol[s1]

            pen = Penalty*min(ap,0.0)^2
            c = R*a + w*earn - ap
            uc = c^(-γ)
            
            np = searchsortedlast(aGrid,ap)
            ##Adjust indices if assets fall out of bounds
            (np > 0 && np < na) ? np = np : 
                (np == na) ? np = na-1 : 
                    np = 1 
            
            ap1,ap2 = aGrid[np],aGrid[np+1]
            basisp1 = (ap2 - ap)/(ap2 - ap1)
            basisp2 = (ap - ap1)/(ap2 - ap1)
            

            ee_rhs = 0.0
            for (sp,earnp) in enumerate(states)
                sp1 = (sp-1)*na + np
                sp2 = (sp-1)*na + np + 1
                
                #Policy
                app = pol_P[sp1]*basisp1 + pol_P[sp2]*basisp2

                cp = R_P*ap + w_P*earnp - app
                ucp = cp^(-γ)

                ###Euler RHS
                ee_rhs += β*(EmpTrans[sp,s]*R_P*ucp + pen)  
            end

            res = uc - ee_rhs
            EE[s1] = res 
        end
    end 

    return EE
end

