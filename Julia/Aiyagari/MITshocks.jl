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

struct Params{T <: Real}
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

mutable struct MitModel{T <: Real,I <: Integer}
    params::Params{T}
    aGrid::Array{T,1} ##Policy grid
    aGridl::Array{T,1}
    na::I ##number of grid points in policy function
    dGrid::Array{T,1}
    nd::I ##number of grid points in distribution
    states::Array{T,1} ##earning states 
    ns::I ##number of states
    EmpTrans::Array{T,2}
end
mutable struct AggVars{T <: Real}
    R::T
    w::T
end

function Prices(K,Z,params::Params)
    @unpack β,α,δ,γ,ρ,σ,lamw,Lbar = params
    R = Z*α*(K/Lbar)^(α-1.0) + 1.0 - δ
    w = Z*(1.0-α)*(K/Lbar)^(1.0-α) 
    
    return AggVars(R,w)
end

function Model(
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
    Penalty::T = 1000000000.0,
    na::I = 201,
    nd::I = 201,
    ns::I = 2,
    endow = [1.0;2.5]) where{T <: Real,I <: Integer}

    #############Params
    params = Params(β,α,δ,γ,ρ,σz,σ,lamw,Lbar,amin,Penalty)
    AggVars = Prices(K,1.0,params)
    @unpack R,w = AggVars

    ################## Policy grid
    function grid_fun(a_min,a_max,na, pexp)
        x = range(a_min,step=0.5,length=na)
        grid = a_min .+ (a_max-a_min)*(x.^pexp/maximum(x.^pexp))
        return grid
    end
    aGrid = grid_fun(amin,amax,na,4.0)
    dGrid=aGrid

    ################## Transition
    EmpTrans = [1.0-lamw σ ;lamw 1.0-σ]
    dis = LinearAlgebra.eigen(EmpTrans)
    mini = argmin(abs.(dis.values .- 1.0)) 
    stdist = abs.(dis.vectors[:,mini]) / sum(abs.(dis.vectors[:,mini]))
    lbar = dot(stdist,endow)
    states = endow/lbar

    @assert sum(EmpTrans[:,1]) == 1.0 ###sum to 1 across rows
    guess = vcat(10.0 .+ aGrid,10.0 .+ aGrid)

    
    return MitModel(params,aGrid,vcat(aGrid,aGrid),na,dGrid,nd,states,ns,EmpTrans),guess,AggVars
end

function interp(x::AbstractArray,
                y::AbstractArray,
                x1::T,
                na::Integer) where{T <: Real}
    
    np = searchsortedlast(x,x1)

    ##Adjust indices if assets fall out of bounds
    (np > 0 && np < na) ? np = np : 
        (np == na) ? np = na-1 : 
            np = 1        
    #@show np
    x_l,x_h = x[np],x[np+1]
    y_l,y_h = y[np], y[np+1] 
    y1 = y_l + (y_h-y_l)/(x_h-x_l)*(x1-x_l) 
    
    above =  y1 > 0.0 
    return above*y1,np
end


function get_c(pol::AbstractArray,
               Pr::AggVars,
               CurrentAssets::AbstractArray,
               MitModel::MitModel,
               cpol::AbstractArray) 
"""
This function gets the consumption implied by savings. Recall that with EGM, you enter with some level of savings (current assets), then you get the implied asset holdings that you enter the period with (pol), this requires finding where each point of CurrentAssets lie in pol, and projecting onto aGrid

######Inputs
pol: 1 dimensional vector of size ns*na - first na elements correspond to 1st idiosyncratic state, and so on 
Pr: type with Interest rate and wage associated with current aggregate states
CurrentAssets: 1 dimensional vector of size ns*na

######Outputs
cpol: consumption vector of size na*ns
"""
    
    @unpack aGrid,na,ns,states = MitModel
    
    pol = reshape(pol,na,ns)
    for si = 1:ns
        for ai = 1:na
            asi = (si - 1)*na + ai
            cpol[asi] = Pr.R*CurrentAssets[asi] + Pr.w*states[si] - interp(pol[:,si],aGrid,CurrentAssets[asi],na)[1]
        end
    end
    return cpol
end

function EulerBack(pol::AbstractArray,
                   Pr::AggVars,
                   Pr_P::AggVars,
                   MitModel::MitModel,
                   cpol::AbstractArray,
                   apol::AbstractArray)

"""
This function updates the policy based on current policy. It uses the Euler equation in a savings problem to get the implied assets holdings and consumption from saving decisions given by the initial grid (aGridl) 

######Inputs
pol: 1 dimensional vector of size ns*na - first na elements correspond to 1st idiosyncratic state, 2nd na to 2nd idiosyncratic state and so on 
Pr: Interest rate and wage associated with current aggregate states today
Pr_p: Interest rate and wage associated with current aggregate states tomorrow
MitModel: holds model type with various objects
cpol: vector to be filled with consumption on the grid
apol: vector to be filled with policy on the grid

######Outputs
apol: policy vector of size na*ns
cpol: consumption vector of size na*ns
"""

    @unpack params,na,nd,ns,aGridl,EmpTrans,states = MitModel
    @unpack γ,β = params

    R_P,w_P = Pr_P.R,Pr_P.w
    R,w = Pr.R,Pr.w
    #cpol = zeros(ns*na)
    #apol = zeros(ns*na)
    cp = get_c(pol,Pr_P,aGridl,MitModel,cpol)
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

function SolveIndProblem(pol::AbstractArray,
                         Pr::AggVars,
                         MitModel::MitModel,
                         cpol::AbstractArray,
                         apol::AbstractArray,tol = 1e-16)
    @unpack ns,na = MitModel

    for i = 1:10000
        a = EulerBack(pol,Pr,Pr,MitModel,cpol,apol)[1]
        if (i-1) % 50 == 0
            test = abs.(a - pol)/(abs.(a) + abs.(pol))
            if maximum(test) < tol
                println("Solved in ",i," ","iterations")
                break
            end
        end
        pol = copy(a)
    end
    return pol
end

function MakeTransMatEGM(pol,MitModel,tmat)
    @unpack ns,na,aGrid,EmpTrans = MitModel
    pol = reshape(pol,na,ns)
    for a_i = 1:na
        for j = 1:ns
            x,i = interp(pol[:,j],aGrid,aGrid[a_i],na)
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

function StationaryDistributionEGM(T)
    N = size(T,1)
    λ, x = powm!(T, rand(N), maxiter = 100000,tol = 1e-11)
    return x/sum(x)
end

function UpdateAggs(
    initialpol::AbstractArray,
    initialdis::AbstractArray,
    Kguess::AbstractArray,
    Zpath::AbstractArray,
    MitModel::MitModel,
    tol = 1e-10,maxn = 50)

    @unpack params,aGridl,na,nd,ns = MitModel

    TimePeriods = length(Kguess)
    tmat = zeros(eltype(initialpol),(na*ns,na*ns))
    cmat = zeros(eltype(initialpol),na*ns)
    cmat2 = zeros(eltype(initialpol),na*ns)
    apols = zeros(eltype(initialpol),na*ns,TimePeriods)
    devol = zeros(eltype(initialpol),na*ns,TimePeriods)
    aggK  = zeros(TimePeriods)

    ##Find policies back through time
    pol = initialpol
    apols[:,TimePeriods] = pol
    for i = TimePeriods:-1:2
        #println("time: ",i-1)
        K,Z_P = Kguess[i],Zpath[i]
        K_m,Z = Kguess[i-1],Zpath[i-1]
        Pr_P = Prices(K,Z_P,params)
        Pr = Prices(K_m,Z,params)
        #cmat .= 0.0
        #cmat2 .= 0.0
        cmat = zeros(eltype(initialpol),na*ns)
        cmat2 = zeros(eltype(initialpol),na*ns)
        pol = EulerBack(pol,Pr,Pr_P,MitModel,cmat,cmat2)[1]
        #pol = SolveIndProblem(pol,Pr,MitModel,cmat,cmat)
        apols[:,i-1] = pol
    end

    #@show apols
    ###Find Evolution of distribution
    dis = initialdis
    aggK[1] = dot(aGridl,dis)
    devol[:,1] = dis
    for i = 1:TimePeriods-1
        #println("time: ",i)
        pol = apols[:,i] ###check on this, I think it was wrong
        tmat .= 0.0
        trans = MakeTransMatEGM(pol,MitModel,tmat)
        dis = trans*dis
        aggK[i+1] = dot(aGridl,dis)
        devol[:,i+1] = dis
    end
        
    return apols,devol,aggK
end

function equilibrium(Kguess,Zpath,initialpol,initialdis,MitModel,tol = 1e-10,maxn = 50)

    pol_ss,dis_ss = initialpol,initialdis
    aggK = copy(Kguess)
    apols,devol,aggK = UpdateAggs(pol_ss,dis_ss,Kguess,Zpath,MitModel)
    #Kguess
    for i = 1:100
        #@show Kguess
        apols,devol,aggK = UpdateAggs(pol_ss,dis_ss,Kguess,Zpath,MitModel)
        @show dif = maximum(abs.(aggK - Kguess))
        if dif < 1e-6
            return apols,devol,aggK
        end
        Kguess = 0.2*aggK + 0.8*Kguess
    end
    #return print("did not converge")
    return apols,devol,aggK
end

####################################################

##############Solve aiyagari economy
## store policies, stationary distribution, and aggregate capital
include("AiyagariEGM.jl")
K0 = 47.0
AA0,pol0,Aggs0 = AiyagariEGM(K0)
pol_ss,_,dis_ss,K_ss,_ = equilibriumEGM(pol0,AA0,K0)

MIT,pol0,Pr0 = Model(K0)
#pol_ss = readdlm("pol_ss.csv")
#dis_ss = readdlm("dis_ss.csv")
#K_ss = dot(dis_ss,AA0.aGridl)
TimePeriods = 200
OffeqPathTime = 1

### steady state productivity
z_ss = 1.0

### Create capital path guess and path of shocks
Kguess = K_ss*ones(TimePeriods+1) #add time 0
Zpath = vcat(1.01*ones(OffeqPathTime),z_ss*ones(TimePeriods+1-OffeqPathTime))

### Solve for transition path back to steady state
#apols,dpols,aggKs = UpdateAggs(pol_ss,dis_ss,Kguess,Zpath,AA0)
eq_apols,eq_dpols,eq_aggKs = equilibrium(Kguess,Zpath,pol_ss,dis_ss,MIT) 


p1 = plot(AA0.aGrid,eq_dpols[1:AA0.na,1])
p1 = plot!(AA0.aGrid,eq_dpols[1:AA0.na,div(TimePeriods,2)])
p1 = plot!(AA0.aGrid,eq_dpols[1:AA0.na,TimePeriods])
p2 = plot(collect(0:1:TimePeriods),eq_aggKs,label="Aggregate capital path from Mit shock")
p = plot(p1,p2, layout=(1,2),size=(1000,400))
savefig(p,"MITfigures.pdf")
#=







####################################################
K0 = 47.0
AA0,pol0,Pr0 = Model(K0)
##predefining matrices
#cMat = fill(0.0,(AA0.na*AA0.ns,))
#aMat = fill(0.0,(AA0.na*AA0.ns,))
#tMat = fill(0.0,(AA0.ns*AA0.nd,AA0.ns*AA0.nd))
pol_ss = readdlm("pol_ss.csv")
dis_ss = readdlm("dis_ss.csv")
K_ss = dot(dis_ss,AA0.aGridl)
TimePeriods = 10
OffeqPathTime = 1

### steady state productivity
z_ss = 1.0

### Create capital path guess and path of shocks
Kguess = K_ss*ones(TimePeriods+1) #add time 0
Zpath = vcat(1.01*ones(OffeqPathTime),z_ss*ones(TimePeriods+1-OffeqPathTime))

### Solve for transition path back to steady state
equilibrium(Kguess,Zpath,pol_ss,dis_ss,AA0)

#=
eq_apols,eq_dpols,eq_aggKs = UpdateAggs(Kguess,Zpath,pol_ss,dis_ss,AA0) 

p1 = plot(AA0.aGrid,eq_dpols[1:AA0.na,1])
p1 = plot!(AA0.aGrid,eq_dpols[1:AA0.na,div(TimePeriods,2)])
p1 = plot!(AA0.aGrid,eq_dpols[1:AA0.na,TimePeriods])
p2 = plot(collect(0:1:TimePeriods),eq_aggKs,label="Aggregate capital path from Mit shock")
p = plot(p1,p2, layout=(1,2),size=(1000,400))
savefig(p,"MITfigures.pdf")



