using LinearAlgebra
using Parameters
using IterativeSolvers
using FastGaussQuadrature
using ForwardDiff
using QuantEcon
using Plots
using Arpack
using BenchmarkTools
include("support.jl")
################################### Model types #########################

uPrime(c,γ) = c.^(-γ)
uPrimeInv(up,γ) = up.^(-1.0/γ)

struct AiyagariParametersEGM{T <: Real}
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
struct AiyagariModelEGM{T <: Real,I <: Integer}
    params::AiyagariParametersEGM{T}
    aGrid::Array{T,1} ##Policy grid
    aGridl::Array{T,1}
    na::I ##number of grid points in policy function
    dGrid::Array{T,1}
    nd::I ##number of grid points in distribution
    states::Array{T,1} ##earning states 
    ns::I ##number of states
    EmpTrans::Array{T,2}
end
mutable struct AggVarsEGM{S <: Real,T <: Real}
    R::S
    w::T
end

function PricesEGM(K,Z,params::AiyagariParametersEGM)
    @unpack β,α,δ,γ,ρ,σ,lamw,Lbar = params
    R = Z*α*(K/Lbar)^(α-1.0) + 1.0 - δ
    w = Z*(1.0-α)*(K/Lbar)^α 
    
    return AggVarsEGM(R,w)
end

function AiyagariEGM(
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
    amin::T = 1e-9,
    amax::T = 200.0,
    Penalty::T = 1000000000.0,
    na::I = 201,
    nd::I = 201,
    ns::I = 2,
    endow = [1.0;2.5]) where{T <: Real,I <: Integer}

    #############Params
    params = AiyagariParametersEGM(β,α,δ,γ,ρ,σz,σ,lamw,Lbar,amin,Penalty)
    AggVars = PricesEGM(K,1.0,params)
    @unpack R,w = AggVars

    ################## Policy grid
    function grid_fun(a_min,a_max,na, pexp)
        x = range(a_min,step=0.5,length=na)
        grid = a_min .+ (a_max-a_min)*(x.^pexp/maximum(x.^pexp))
        return grid
    end
    aGrid = grid_fun(amin,amax,na,4.0)
    ################### Distribution grid
    #dGrid = collect(range(aGrid[1],stop = aGrid[end],length = nd))
    #aGrid = collect(range(amin,stop = amax,length = na))
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
    
    guess = vcat(10.0 .+ aGrid,10.0 .+ aGrid)

    
    return AiyagariModelEGM(params,aGrid,vcat(aGrid,aGrid),na,dGrid,nd,states,ns,EmpTrans),guess,AggVars
end

function interpEGM(pol::AbstractArray,
                grid::AbstractArray,
                x::T,
                na::Integer) where{T <: Real}
    np = searchsortedlast(pol,x)

    ##Adjust indices if assets fall out of bounds
    (np > 0 && np < na) ? np = np : 
        (np == na) ? np = na-1 : 
            np = 1        
    #@show np
    ap_l,ap_h = pol[np],pol[np+1]
    a_l,a_h = grid[np], grid[np+1] 
    ap = a_l + (a_h-a_l)/(ap_h-ap_l)*(x-ap_l) 
    
    above =  ap > 0.0 
    return above*ap,np
end

function get_cEGM(pol::AbstractArray,
               Aggs::AggVarsEGM,
               CurrentAssets::AbstractArray,
               AiyagariModel::AiyagariModelEGM,
               cpol::AbstractArray) 
    
    @unpack aGrid,na,ns,states = AiyagariModel
    pol = reshape(pol,na,ns)
    for si = 1:ns
        for ai = 1:na
            asi = (si - 1)*na + ai
            cpol[asi] = Aggs.R*CurrentAssets[asi] + Aggs.w*states[si] - interpEGM(pol[:,si],aGrid,CurrentAssets[asi],na)[1]
        end
    end
    return cpol
end

function EulerBackEGM(pol::AbstractArray,
                   Aggs::AggVarsEGM,
                   Aggs_P::AggVarsEGM,
                   AiyagariModel::AiyagariModelEGM,
                   cpol::AbstractArray,
                   apol::AbstractArray)
    
    @unpack params,na,nd,ns,aGridl,EmpTrans,states = AiyagariModel
    @unpack γ,β = params

    R_P,w_P = Aggs_P.R,Aggs_P.w
    R,w = Aggs.R,Aggs.w
    
    cp = get_cEGM(pol,Aggs_P,aGridl,AiyagariModel,cpol)
    upcp = uPrime(cp,γ)
    Eupcp = copy(cpol)
    #Eupcp_sp = 0.0

    for ai = 1:na
        for si = 1:ns
            asi = (si-1)*na + ai
            Eupcp_sp = 0.0
            for spi = 1:ns
                aspi = (spi-1)*na + ai
                Eupcp_sp += EmpTrans[spi,si]*upcp[aspi]
            end
            Eupcp[asi] = Eupcp_sp 
        end
    end

    upc = R_P*β*Eupcp

    c = uPrimeInv(upc,γ)

    for ai = 1:na
        for si = 1:ns
            asi = (si-1)*na+ai
            apol[asi] = (aGridl[asi] + c[asi] - w*states[si])/R
        end
    end

    return apol,c
end


function SolveEGM(pol::AbstractArray,
                  Aggs::AggVarsEGM,
                  AiyagariModel::AiyagariModelEGM,
                  cpol::AbstractArray,
                  apol::AbstractArray,tol = 1e-17)
    @unpack ns,na = AiyagariModel

    for i = 1:10000
        a = EulerBackEGM(pol,Aggs,Aggs,AiyagariModel,cpol,apol)[1]
        if (i-1) % 50 == 0
            test = abs.(a - pol)/(abs.(a) + abs.(pol))
            #println("iteration: ",i," ",maximum(test))
            if maximum(test) < tol
                println("Solved in ",i," ","iterations")
                break
            end
        end
        pol = copy(a)
    end
    return pol
end

function MakeTransMatEGM(pol,AiyagariModel,tmat)
    @unpack ns,na,aGrid,EmpTrans = AiyagariModel
    pol = reshape(pol,na,ns)
    for a_i = 1:na
        for j = 1:ns
            x,i = interpEGM(pol[:,j],aGrid,aGrid[a_i],na)
            p = (aGrid[a_i] - pol[i,j])/(pol[i+1,j] - pol[i,j])
            p = min(max(p,0.0),1.0)
            sj = (j-1)*na
            for k = 1:ns
                sk = (k-1)*na
                tmat[sk+i+1,sj+a_i] = p * EmpTrans[k,j]
                tmat[sk+i,sj+a_i] = (1.0-p) * EmpTrans[k,j]
            end
        end
    end
    return tmat
end


function MakeEmpTransEGM(AiyagariModel)
    @unpack params,nd,EmpTrans = AiyagariModel
    δ = params.δ
    eye = LinearAlgebra.eye(eltype(EmpTrans),nd)
    hcat(vcat(eye*EmpTrans[1,1],eye*EmpTrans[2,1]),vcat(eye*EmpTrans[1,2],eye*EmpTrans[2,2]))
end

function StationaryDistributionEGM(T,AiyagariModel)
    @unpack ns,nd = AiyagariModel 
    λ, x = powm!(T, rand(ns*nd), maxiter = 100000,tol = 1e-15)
    return x/sum(x)
end

function equilibriumEGM(
    initialpol::AbstractArray,
    AiyagariModel::AiyagariModelEGM,
    K0::T,
    tol = 1e-10,maxn = 50)where{T <: Real}

    @unpack params,aGrid,na,dGrid,nd,ns = AiyagariModel

    EA = 0.0

    tmat = zeros(eltype(initialpol),(na*ns,na*ns))
    cmat = zeros(eltype(initialpol),na*ns)

    ###Start Bisection
    uK,lK = K0, 0.0
    
    pol = initialpol
    #uir,lir = initialR, 1.0001
    print("Iterate on aggregate assets")
    for kit = 1:maxn
        Aggs = PricesEGM(K0,1.0,params) ##steady state
        cmat .= 0.0
        pol = SolveEGM(pol,Aggs,AiyagariModel,cmat,cmat)

        #Stationary transition
        tmat .= 0.0
        trans = MakeTransMatEGM(pol,AiyagariModel,tmat)
        D = StationaryDistributionEGM(trans,AiyagariModel)

        #Aggregate savings
        EA = dot(vcat(dGrid,dGrid),D)
        
        if (EA > K0) ### too little lending -> low r -> too much borrowing 
            uK = min(EA,uK)  
            lK = max(K0,lK)
            K0 = 1.0/2.0*(lK + uK)
        else ## too much lending -> high r -> too little borrowing
            uK = min(K0,uK)
            lK = max(EA,lK)
            K0 = 1.0/2.0*(lK + uK)
        end
        println("Interest rate: ",Aggs.R," ","Bond Supply: ",EA)
        #@show K0
        if abs(EA - K0) < 1e-7
            println("Markets clear!")
            #println("Interest rate: ",R," ","Bonds: ",EA)
            cmat .= 0.0
            polA,polC = EulerBackEGM(pol,Aggs,Aggs,AiyagariModel,cmat,cmat)
            return polA,polC,D,EA,Aggs
            break
        end
    end
    
    return println("Markets did not clear")
end

function EulerResidualEGM(pol::AbstractArray,
                       pol_P::AbstractArray,
                       Aggs::AggVarsEGM,
                       Aggs_P::AggVarsEGM,
                       AiyagariModel::AiyagariModelEGM,
                       cmat::AbstractArray,
                          amat::AbstractArray)
    
    a,c = EulerBackEGM(pol_P,Aggs,Aggs_P,AiyagariModel,cmat,amat)
    c2 = get_cEGM(pol,Aggs,a,AiyagariModel,cmat)

    return (c ./ c2 .- 1.0)
end


function WealthResidualEGM(pol::AbstractArray,
                        D_L::AbstractArray,
                        D::AbstractArray,
                        AiyagariModel::AiyagariModelEGM,
                        tmat::AbstractArray)
    return (D - MakeTransMatEGM(pol,AiyagariModel,tmat) * D_L)[2:end]
end

function AggResidual(D::AbstractArray,K,Z_L,Z,epsilon::AbstractArray,AiyagariModel::AiyagariModelEGM)
    @unpack params,dGrid = AiyagariModel
    @unpack ρ,σz = params
    ϵz = epsilon[1]

    AggAssets = dot(D,vcat(dGrid,dGrid))
    AggEqs = vcat(
        AggAssets - K, #bond market clearing
        log(Z) - ρ*log(Z_L) - ϵz, #TFP evol
    ) 
    
    return AggEqs
end

function FEGM(X_L::AbstractArray,
           X::AbstractArray,
           X_P::AbstractArray,
           epsilon::AbstractArray,
           AiyagariModel::AiyagariModelEGM,
           pos)
    
    @unpack params,na,ns,nd,dGrid = AiyagariModel

    
    m = na*ns
    md = nd*ns
    pol_L,D_L,Agg_L = X_L[1:m],X_L[m+1:m+md-1],X_L[m+md:end]
    pol,D,Agg = X[1:m],X[m+1:m+md-1],X[m+md:end]
    pol_P,D_P,Agg_P = X_P[1:m],X_P[m+1:m+md-1],X_P[m+md:end]

    K_L,Z_L = Agg_L
    K,Z = Agg
    K_P,Z_P = Agg_P

    D_L = vcat(1.0-sum(D_L),D_L)
    D   = vcat(1.0-sum(D),D)
    D_P = vcat(1.0-sum(D_P),D_P)
    
    
    Price = PricesEGM(K_L,Z,params)
    Price_P = PricesEGM(K,Z_P,params)
    #@show Price_P
    #Need matrices that pass through intermediate functions to have the same type as the
    #argument of the derivative that will be a dual number when using forward diff. In other words,
    #when taking derivative with respect to X_P, EE, his, his_rhs must have the same type as X_P
    if pos == 1 
        cmat = zeros(eltype(X_L),na*ns)
        cmat2 = zeros(eltype(X_L),na*ns)
        tmat = zeros(eltype(X_L),(ns*na,ns*na))
    elseif pos == 2
        cmat = zeros(eltype(X),na*ns)
        cmat2 = zeros(eltype(X),na*ns)
        tmat = zeros(eltype(X),(ns*na,ns*na))
    else
        cmat = zeros(eltype(X_P),na*ns)
        cmat2 = zeros(eltype(X_P),na*ns)
        tmat = zeros(eltype(X_P),(ns*na,ns*na))
    end
    agg_root = AggResidual(D,K,Z_L,Z,epsilon,AiyagariModel)
    dist_root = WealthResidualEGM(pol,D_L,D,AiyagariModel,tmat) ###Price issue
    euler_root = EulerResidualEGM(pol,pol_P,Price,Price_P,AiyagariModel,cmat,cmat2)

    return vcat(euler_root,dist_root,agg_root)
end

function EulerBackError(pol::AbstractArray,
                   Aggs::AggVarsEGM,
                   Aggs_P::AggVarsEGM,
                   AiyagariModel::AiyagariModelEGM,
                   cpol::AbstractArray,
                   apol::AbstractArray,
                   Grid::AbstractArray)
    
    @unpack params,na,nd,ns,aGridl,states,EmpTrans = AiyagariModel
    @unpack γ,β = params

    R_P,w_P = Aggs_P.R,Aggs_P.w
    R,w = Aggs.R,Aggs.w
    ng = div(length(Grid),ns)
    cp = get_c_error(pol,Aggs_P,Grid,AiyagariModel,cpol)
    upcp = uPrime(cp,γ)
    Eupcp = copy(cpol)

    for ai = 1:ng
        for si = 1:ns
            asi = (si-1)*ng + ai
            Eupcp_sp = 0.0
            for spi = 1:ns
                aspi = (spi-1)*ng + ai
                Eupcp_sp += EmpTrans[spi,si]*upcp[aspi]
            end
            Eupcp[asi] = Eupcp_sp 
        end
    end

    upc = R_P*β*Eupcp

    c = uPrimeInv(upc,γ)

    for ai = 1:ng
        for si = 1:ns
            asi = (si-1)*ng+ai
            apol[asi] = (Grid[asi] + c[asi] - w*states[si])/R
        end
    end

    return apol,c
end
function get_c_error(pol::AbstractArray,
               Aggs::AggVarsEGM,
               CurrentAssets::AbstractArray,
               AiyagariModel::AiyagariModelEGM,
               cpol::AbstractArray) 
    
    @unpack aGrid,na,ns,states = AiyagariModel
    
    pol = reshape(pol,na,ns)
    Gsize = div(length(CurrentAssets),ns)
    for si = 1:ns
        for ai = 1:Gsize
            asi = (si - 1)*Gsize + ai
            cpol[asi] = Aggs.R*CurrentAssets[asi] + Aggs.w*states[si] - interpEGM(pol[:,si],aGrid,CurrentAssets[asi],na)[1]
        end
    end
    return cpol
end

function EulerResidualError(pol::AbstractArray,
                       pol_P::AbstractArray,
                       Aggs::AggVarsEGM,
                       Aggs_P::AggVarsEGM,
                       AiyagariModel::AiyagariModelEGM,
                       cmat::AbstractArray,
                       amat::AbstractArray,
                       Grid::AbstractArray)    
    
    a,c = EulerBackError(pol_P,Aggs,Aggs_P,AiyagariModel,cmat,amat,Grid)
    c2 = get_c_error(pol,Aggs,a,AiyagariModel,cmat)
    return (c ./ c2 .- 1.0)
end



