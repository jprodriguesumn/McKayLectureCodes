using LinearAlgebra
using Parameters
using IterativeSolvers
using FastGaussQuadrature
using ForwardDiff
using BenchmarkTools
#using Calculus
using Plots
using Arpack
using QuantEcon
include("support.jl")
################################### Model types #########################

struct MarketParametersEGM{T <: Real}
    β::T
    γ::T
    ρz::T
    σz::T
    ρξ::T
    σξ::T
    ζ::T
    ψ::T
    μ::T
    μϵ::T
    θ::T
    ω::T
    ubar::T
    δ::T
    Mbar::T
    wbar::T
    B::T
    ben::T
    amin::T
    Penalty::T
end

mutable struct AggVarsEGM{T <: Real,S <: Real,D <: Real}
    R::T
    M::T
    w::T
    Earnings::Array{S,1}
    EmpTrans::Array{D,2}
end

struct HankModelEGM{T <: Real,I <: Integer}
    params::MarketParametersEGM{T}
    aGrid::Array{T,1}
    aGridl::Array{T,1}
    na::I
    dGrid::Array{T,1}
    nd::I
    ns::I
end

function PricesEGM(R,M,Z,u,ulag,params::MarketParametersEGM)
    @unpack δ,Mbar,wbar,ψ,ben,B,ζ = params
    w = wbar * (M/Mbar)^ζ
    Y = Z*(1.0-u)
    H = (1.0-u) - (1.0 - δ)*(1.0-ulag)
    d = (Y-ψ*M*H)/(1.0-u) - w
    τ = ((R-1.0)*B + ben*u)/(w+d)/(1.0-u)
    EmpInc = (1.0-τ)*(w+d)
    Earnings = vcat(ben,EmpInc)
    EmpTrans = vcat(hcat(1.0-M,δ*(1.0-M)),hcat(M,1.0-δ*(1.0-M)))

    return AggVarsEGM(R,M,w,Earnings,EmpTrans)
end

uPrime(c,γ) = c.^(-γ)
uPrimeInv(up,γ) = up.^(-1.0/γ)

function HankEGM(
    R::T,
    β::T = 0.98,
    γ::T = 2.00,
    ρz::T = 0.95,
    σz::T = 1.0,
    ρξ::T = 0.9,
    σξ::T = 1.0,
    ζ::T = 0.8,
    ψ::T = 0.1,
    μ::T = 1.2,
    θ::T = 0.75,
    ω::T = 1.5,
    ubar::T = 0.15,
    δ::T = 0.06,
    B::T = 1.0,
    amin::T = 1e-10,
    amax::T = 200.0,
    na::I = 201,
    nd::I = 201,
    Penalty::T = 10000000000000000.0) where{T <: Real,I <: Integer}

    #############Params
    μϵ = μ/(μ-1.0)
    Mbar = (1.0-ubar)*δ/(ubar+δ*(1.0-ubar))
    wbar = 1.0/μ - δ*ψ*Mbar
    ben = wbar*0.5
    params = MarketParametersEGM(β,γ,ρz,σz,ρξ,σξ,ζ,ψ,μ,μϵ,θ,ω,ubar,δ,Mbar,wbar,B,ben,amin,Penalty)
    AggVars = PricesEGM(R,Mbar,1.0,ubar,ubar,params)
    @unpack R,M,w,Earnings,EmpTrans = AggVars
    ################## Collocation pieces
    function grid_fun(a_min,a_max,na, pexp)
        x = range(a_min,step=0.5,length=na)
        grid = a_min .+ (a_max-a_min)*(x.^pexp/maximum(x.^pexp))
        return grid
    end
    aGrid = grid_fun(amin,amax,na,4.0)
    #Collocation = ReiterCollocation(aGrid,aSize)

    ################### Distribution pieces
    #dGrid = collect(range(aGrid[1],stop = 10.0,length = nd))
    dGrid = aGrid
    ns = length(Earnings)
    guess = vcat(10.0 .+ aGrid,10.0 .+ aGrid)
    cmat = fill(0.0,(na,ns))
    tmat = fill(0.0,(na*ns,na*ns))
    
    return HankModelEGM(params,aGrid,vcat(aGrid,aGrid),na,dGrid,nd,ns),guess,AggVars
end

#function interp(x,y,x1)
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
               HankModel::HankModelEGM,
               cpol::AbstractArray) 
    
    @unpack aGrid,na,ns = HankModel
    Earnings = Aggs.Earnings
    
    pol = reshape(pol,na,ns)
    #Gsize = div(length(CurrentAssets),ns)
    for si = 1:ns
        for ai = 1:na
            asi = (si - 1)*na + ai
            cpol[asi] = Aggs.R*CurrentAssets[asi] + Earnings[si] - interpEGM(pol[:,si],aGrid,CurrentAssets[asi],na)[1]
        end
    end
    return cpol
end

function EulerBackEGM(pol::AbstractArray,
                   Aggs::AggVarsEGM,
                   Aggs_P::AggVarsEGM,
                   HankModel::HankModelEGM,
                   cpol::AbstractArray,
                   apol::AbstractArray)
    
    @unpack params,na,nd,ns,aGridl = HankModel
    @unpack γ,β = params

    EmpTrans_P,R_P = Aggs_P.EmpTrans,Aggs_P.R
    Earnings,R = Aggs.Earnings,Aggs.R
    
    cp = get_cEGM(pol,Aggs_P,aGridl,HankModel,cpol)
    upcp = uPrime(cp,γ)
    Eupcp = copy(cpol)
    #Eupcp_sp = 0.0

    for ai = 1:na
        for si = 1:ns
            asi = (si-1)*na + ai
            Eupcp_sp = 0.0
            for spi = 1:ns
                aspi = (spi-1)*na + ai
                Eupcp_sp += EmpTrans_P[spi,si]*upcp[aspi]
            end
            Eupcp[asi] = Eupcp_sp 
        end
    end

    upc = R_P*β*Eupcp

    c = uPrimeInv(upc,γ)

    for ai = 1:na
        for si = 1:ns
            asi = (si-1)*na+ai
            apol[asi] = (aGridl[asi] + c[asi] - Earnings[si])/R
        end
    end

    return apol,c
end

function SolveEGM(pol::AbstractArray,
                  Aggs::AggVarsEGM,
                  HankModel::HankModelEGM,
                  cpol::AbstractArray,
                  apol::AbstractArray,tol = 1e-16)
    @unpack ns,na = HankModel

    for i = 1:10000
        a = EulerBackEGM(pol,Aggs,Aggs,HankModel,cpol,apol)[1]
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

function MakeSavTransEGM(pol,HankModel,tmat)
    @unpack ns,na,aGrid = HankModel
    pol = reshape(pol,na,ns)
    for a_i = 1:na
        for s_j = 1:ns
            x,i = interpEGM(pol[:,s_j],aGrid,aGrid[a_i],na)
            p = (aGrid[a_i] - pol[i,s_j])/(pol[i+1,s_j] - pol[i,s_j])
            p = min(max(p,0.0),1.0)
            ss = (s_j-1)*na
            tmat[ss+i+1,ss+a_i] = p
            tmat[ss+i,ss+a_i] = (1.0-p)              
        end
    end
    return tmat
end


function MakeEmpTransEGM(M,HankModel)
    @unpack params,nd = HankModel
    δ = params.δ
    #eye = LinearAlgebra.eye(eltype(M),nd)
    eye = Matrix{eltype(M)}(I,nd,nd)
    hcat(vcat(eye*(1.0-M),eye*M),vcat(eye*δ*(1.0-M),eye*(1.0-δ*(1.0-M))))
end

function MakeTransMatEGM(pol,M,HankModel,tmat)
    MakeSavTransEGM(pol,HankModel,tmat)*MakeEmpTransEGM(M,HankModel)
end

function StationaryDistributionEGM(T,HankModel)
    @unpack ns,nd = HankModel 
    λ, x = powm!(T, rand(ns*nd), maxiter = 100000,tol = 1e-15)
    #@show λ
    return x/sum(x)
end

function equilibriumEGM(
    initialpol::AbstractArray,
    HankModel::HankModelEGM,
    initialR::T,
    tol = 1e-10,maxn = 200)where{T <: Real}

    @unpack params,aGrid,na,dGrid,nd,ns = HankModel
    @unpack B,ubar,Mbar,Penalty = params

    EA = 0.0

    tmat = zeros(eltype(initialpol),(na*ns,na*ns))
    cmat = zeros(eltype(initialpol),na*ns)
    ###Start Bisection
    K = 0.0
    R = initialR
    Aggs = PricesEGM(R,Mbar,1.0,ubar,ubar,params)
    pol = initialpol
    uir,lir = initialR, 1.0001
    print("Iterate on aggregate assets")
    for kit = 1:maxn
        Aggs = PricesEGM(R,Mbar,1.0,ubar,ubar,params) ##steady state
        cmat .= 0.0
        pol = SolveEGM(pol,Aggs,HankModel,cmat,cmat)

        #Stationary transition
        tmat .= 0.0
        trans = MakeTransMatEGM(pol,Aggs.M,HankModel,tmat)
        D = StationaryDistributionEGM(trans,HankModel)

        #Aggregate savings
        EA = dot(vcat(dGrid,dGrid),D)
        
        if (EA > B)
            uir = min(uir,R)
            R = 1.0/2.0*(uir + lir)
        else
            lir = max(lir,R)
            R = 1.0/2.0*(uir + lir)
        end
        println("Interest rate: ",R," ","Bond Supply: ",EA)
        if abs(EA - B) < 1e-10
            println("Markets clear!")
            #println("Interest rate: ",R," ","Bonds: ",EA)
            cmat .= 0.0
            polA,polC = EulerBackEGM(pol,Aggs,Aggs,HankModel,cmat,cmat)
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
                       HankModel::HankModelEGM,
                       cmat::AbstractArray,
                       amat::AbstractArray)
    
    a,c = EulerBackEGM(pol_P,Aggs,Aggs_P,HankModel,cmat,amat)
    c2 = get_cEGM(pol,Aggs,a,HankModel,cmat)
    return (c ./ c2 .- 1.0)
end

function WealthResidualEGM(pol::AbstractArray,
                        D_L::AbstractArray,
                        D::AbstractArray,
                        M::T,
                        HankModel::HankModelEGM,
                        tmat::AbstractArray) where{T <: Real}
    return (D - MakeTransMatEGM(pol,M,HankModel,tmat) * D_L)[2:end]
end

function AggResidual(D::AbstractArray,u,u_L,R,i_L,i,M,M_P,pi,pi_P,pA,pB,pA_P,pB_P,Z_L,Z,ξ_L,ξ,Rstar,epsilon::AbstractArray,HankModel::HankModelEGM)where{T <: Real}

    @unpack params,dGrid = HankModel
    @unpack β,γ,ρz,σz,ρξ,σξ,ζ,ψ,μ,μϵ,θ,ω,ubar,δ,Mbar,wbar,B,ben,amin = params
    ϵz,ϵξ = epsilon
    #@show D[1:30]
    AggAssets = dot(D,vcat(dGrid,dGrid))
    Y = Z*(1.0-u)
    H = 1.0-u - (1.0-δ)*(1.0-u_L)
    marg_cost = (wbar * (M/Mbar)^ζ + ψ*M - (1.0-δ)*ψ*M_P)/Z
    AggEqs = vcat(
        AggAssets - B, #bond market clearing
        1.0 + i - Rstar * pi^ω * ξ, #mon pol rule #notice Rstar here is fixed and defined as a global
        R - (1.0 + i_L)/pi, 
        M - (1.0-u-(1.0-δ)*(1.0-u_L))/(u_L + δ*(1.0-u_L)), #3 labor market 
        pi - θ^(1.0/(1.0-μϵ))*(1.0-(1.0-θ)*(pA/pB)^(1.0-μϵ))^(1.0/(μϵ-1.0)), #4 inflation
        -pA + μ*Y*marg_cost + θ*pi_P^μϵ*pA_P/R, #aux inflation equ 1
        -pB + Y + θ*pi_P^(μϵ-1.0)*pB_P/R, #aux inflation equ 2
        log(Z) - ρz*log(Z_L) - σz*ϵz, #TFP evol
        log(ξ) - ρξ*log(ξ_L) - σξ*ϵξ #Mon shock
    ) 
    
    return AggEqs
end

function FEGM(X_L::AbstractArray,
           X::AbstractArray,
           X_P::AbstractArray,
           epsilon::AbstractArray,
           HankModel::HankModelEGM,
           AggsSS::AggVarsEGM,
           pos)
    
    @unpack params,na,ns,nd,dGrid = HankModel
    Rstar = AggsSS.R

    
    m = na*ns
    md = nd*ns
    pol_L,D_L,Agg_L = X_L[1:m],X_L[m+1:m+md-1],X_L[m+md:end]
    pol,D,Agg = X[1:m],X[m+1:m+md-1],X[m+md:end]
    pol_P,D_P,Agg_P = X_P[1:m],X_P[m+1:m+md-1],X_P[m+md:end]

    u_L, R_L, i_L, M_L, pi_L, pA_L, pB_L, Z_L, ξ_L = Agg_L
    u, R, i, M, pi, pA, pB, Z, ξ = Agg
    u_P, R_P, i_P, M_P, pi_P, pA_P, pB_P, Z_P, ξ_P = Agg_P
    
    D_L = vcat(1.0-sum(D_L),D_L)
    D   = vcat(1.0-sum(D),D)
    D_P = vcat(1.0-sum(D_P),D_P)
    
    
    Price = PricesEGM(R,M,Z,u,u_L,params)
    Price_P = PricesEGM(R_P,M_P,Z_P,u_P,u,params)
    #typeof(Price)
    #typeof(Price_P)
    #Need matrices that pass through intermediate functions to have the same type as the
    #argument of the derivative that will be a dual number when using forward diff. In other words,
    #when taking derivative with respect to X_P, EE, his, his_rhs must have the same type as X_P
    if pos == 1 
        cmat = zeros(eltype(X_L),na*ns)
        tmat = zeros(eltype(X_L),(ns*na,ns*na))
    elseif pos == 2
        cmat = zeros(eltype(X),na*ns)
        tmat = zeros(eltype(X),(ns*na,ns*na))
    else
        cmat = zeros(eltype(X_P),na*ns)
        tmat = zeros(eltype(X_P),(ns*na,ns*na))
    end
    
    agg_root = AggResidual(D,u,u_L,Price.R,i_L,i,M,M_P,pi,pi_P,pA,pB,pA_P,pB_P,Z_L,Z,ξ_L,ξ,Rstar,epsilon,HankModel)
    dist_root = WealthResidualEGM(pol,D_L,D,Price.M,HankModel,tmat) ###Price issue
    euler_root = EulerResidualEGM(pol,pol_P,Price,Price_P,HankModel,cmat,cmat)
    
    return vcat(euler_root,dist_root,agg_root)
end

function EulerBackError(pol::AbstractArray,
                   Aggs::AggVarsEGM,
                   Aggs_P::AggVarsEGM,
                   HankModel::HankModelEGM,
                   cpol::AbstractArray,
                   apol::AbstractArray,
                   Grid::AbstractArray)
    
    @unpack params,na,nd,ns,aGridl = HankModel
    @unpack γ,β = params

    EmpTrans_P,R_P = Aggs_P.EmpTrans,Aggs_P.R
    Earnings,R = Aggs.Earnings,Aggs.R
    ng = div(length(Grid),ns)
    cp = get_c_error(pol,Aggs_P,Grid,HankModel,cpol)
    upcp = uPrime(cp,γ)
    Eupcp = copy(cpol)
    #Eupcp_sp = 0.0

    for ai = 1:ng
        for si = 1:ns
            asi = (si-1)*ng + ai
            Eupcp_sp = 0.0
            for spi = 1:ns
                aspi = (spi-1)*ng + ai
                Eupcp_sp += EmpTrans_P[spi,si]*upcp[aspi]
            end
            Eupcp[asi] = Eupcp_sp 
        end
    end

    upc = R_P*β*Eupcp

    c = uPrimeInv(upc,γ)

    for ai = 1:ng
        for si = 1:ns
            asi = (si-1)*ng+ai
            apol[asi] = (Grid[asi] + c[asi] - Earnings[si])/R
        end
    end

    return apol,c
end
function get_c_error(pol::AbstractArray,
               Aggs::AggVarsEGM,
               CurrentAssets::AbstractArray,
               HankModel::HankModelEGM,
               cpol::AbstractArray) 
    
    @unpack aGrid,na,ns = HankModel
    Earnings = Aggs.Earnings
    
    pol = reshape(pol,na,ns)
    @show Gsize = div(length(CurrentAssets),ns)
    for si = 1:ns
        for ai = 1:Gsize
            asi = (si - 1)*Gsize + ai
            cpol[asi] = Aggs.R*CurrentAssets[asi] + Earnings[si] - interpEGM(pol[:,si],aGrid,CurrentAssets[asi],na)[1]
        end
    end
    return cpol
end

function EulerResidualError(pol::AbstractArray,
                       pol_P::AbstractArray,
                       Aggs::AggVarsEGM,
                       Aggs_P::AggVarsEGM,
                       HankModel::HankModelEGM,
                       cmat::AbstractArray,
                       amat::AbstractArray,
                       Grid::AbstractArray)    
    
    a,c = EulerBackError(pol_P,Aggs,Aggs_P,HankModel,cmat,amat,Grid)
    c2 = get_c_error(pol,Aggs,a,HankModel,cmat)
    return (c ./ c2 - 1.0)
end


R0 = 1.01
HM0,pol0,Aggs0 = HankEGM(R0)

##predefining matrices
cMat = fill(0.0,(HM0.na*HM0.ns,))
aMat = fill(0.0,(HM0.na*HM0.ns,))
transMat = fill(0.0,(HM0.ns*HM0.nd,HM0.ns*HM0.nd))

###testing forwarddiff
#@show interp(pol0[1:HM0.na],HM0.aGrid,2.0,HM0.na)
#testd = ForwardDiff.gradient(t -> interp(t,HM0.aGrid,2.0,HM0.na)[1],pol0[:,1])
#eutest = ForwardDiff.jacobian(t -> EulerBack(t,Aggs0,Aggs0,HM0)[1],pol0)
#get_c(pol0,Aggs0,HM0.aGridl,HM0,cMat)
#eua,euc =  EulerBack(pol0,Aggs0,Aggs0,HM0,cMat,aMat)
#pol =SolveEGM(pol0,Aggs0,HM0,cMat,aMat)
#=fineGridl = vcat(collect(range(HM0.aGrid[1],stop = HM0.aGrid[end],length = 10000)),collect(range(HM0.aGrid[1],stop = HM0.aGrid[end],length = 10000)))
fineGridzero = fill(0.0,HM0.ns*10000)
a0,c0 = EulerBack(pol,Aggs0,Aggs0,HM0,cMat,aMat)
err = EulerResidualError(pol,pol,Aggs0,Aggs0,HM0,fineGridzero,fineGridzero,fineGridl)
a,c = EulerBackError(pol,Aggs0,Aggs0,HM0,fineGridzero,fineGridzero,fineGridl)
errm = reshape(err,10000,HM0.ns)
@show sum(abs.(errm[:,1]))
@show sum(abs.(errm[:,2]))
p1 = plot(fineGridl[1:10000],errm[:,1])
p2 = plot(fineGridl[1:10000],errm[:,2])
p = plot(p1,p2, layout=(1,2))
savefig(p,"errEGM.pdf")=#



#c2 = get_c(pol,Aggs0,a,HM0,fineGridzero)

#polA_ss,polC_ss,D_ss,K_ss,Aggs_ss = equilibriumEGM(pol0,HM0,1.01)
#=

#polA_ss,polC_ss,D_ss,K_ss,Aggs_ss = equilibrium(pol0,HM0,1.01)

pB = (1.0-HM0.params.ubar)/(1.0-HM0.params.θ/Aggs_ss.R)
Agg_ss = [HM0.params.ubar;Aggs_ss.R;Aggs_ss.R-1.0;HM0.params.Mbar;1.0;pB;pB;1.0;1.0]
xss = vcat(reshape(polA_ss,(HM0.ns*HM0.na,)),D_ss[2:end],Agg_ss)

F(xss,xss,xss,[0.0;0.0],HM0,Aggs_ss,3)

Am = ForwardDiff.jacobian(t -> F(xss,xss,t,[0.0;0.0],HM0,Aggs_ss,3),xss)
Bm = ForwardDiff.jacobian(t -> F(xss,t,xss,[0.0;0.0],HM0,Aggs_ss,2),xss)
Cm = ForwardDiff.jacobian(t -> F(t,xss,xss,[0.0;0.0],HM0,Aggs_ss,1),xss)
Em = ForwardDiff.jacobian(t -> F(xss,xss,xss,t,HM0,Aggs_ss,1),[0.0;0.0])


@time P,Q = SolveSystem(Am,Bm,Cm,Em)



simul_length = 200
H = Matrix(I,size(P,1),size(P,1))
lss = LSS(P,Q,H)
X_simulH, _ = simulate(lss, simul_length);

Time = 100
IRFZ = fill(0.0,(length(xss),Time))
IRFZ[:,1] = Q[:,1]
for t =2:Time
    IRFZ[:,t] = P*IRFZ[:,t-1]
end
IRFxi = fill(0.0,(length(xss),Time))
IRFxi[:,1] = Q[:,2]
for t =2:Time
    IRFxi[:,t] = P*IRFxi[:,t-1]
end


include("RankJoao.jl")
na = HM0.na
ns = HM0.ns
nx = HM0.nd

p1 = plot(IRFZ[ns*na + ns*nx,:],title="u",label="Hank",titlefont=font(7, "Courier"))
p1 = plot!(IRFRankZ[1,:],title="u",label="Rank",titlefont=font(7, "Courier"))

p2 = plot(IRFZ[ns*na + ns*nx+1,:],title="R",label="Hank",titlefont=font(7, "Courier"))
p2 = plot!(IRFRankZ[2,:],title="R",label="Rank",titlefont=font(7, "Courier"))

p3 = plot(IRFZ[ns*na + ns*nx+2,:],title="i",label="Hank",titlefont=font(7, "Courier"))
p3 = plot!(IRFRankZ[11,:],title="i",label="Rank",titlefont=font(7, "Courier"))

p4 = plot(IRFZ[ns*na + ns*nx+3,:],title="M",label="Hank",titlefont=font(7, "Courier"))
p4 = plot!(IRFRankZ[3,:],title="M",label="Rank",titlefont=font(7, "Courier"))

p5 = plot(IRFZ[ns*na + ns*nx+4,:],title="pi",label="Hank",titlefont=font(7, "Courier"))
p5 = plot!(IRFRankZ[4,:],title="pi",label="Rank",titlefont=font(7, "Courier"))

p6 = plot(IRFZ[ns*na + ns*nx+5,:],title="pA",label="Hank",titlefont=font(7, "Courier"))
p6 = plot!(IRFRankZ[5,:],title="pA",label="Rank",titlefont=font(7, "Courier"))

p7 = plot(IRFZ[ns*na + ns*nx+6,:],title="pB",label="Hank",titlefont=font(7, "Courier"))
p7 = plot!(IRFRankZ[6,:],title="pB",label="Rank",titlefont=font(7, "Courier"))

p8 = plot(IRFZ[ns*na + ns*nx+7,:],title="Z",label="Hank",titlefont=font(7, "Courier"))
p8 = plot!(IRFRankZ[7,:],title="Z",label="Rank",titlefont=font(7, "Courier"))

p = plot(p1,p2,p3,p4,p5,p6,p7,p8, layout=(4,2), size=(1600,700))
savefig(p,"z_irfs.pdf")

p1 = plot(IRFxi[ns*na + ns*nx,:],title="u",label="Hank",titlefont=font(7, "Courier"))
p1 = plot!(IRFRankxi[1,:],title="u",label="Rank",titlefont=font(7, "Courier"))

p2 = plot(IRFxi[ns*na + ns*nx+1,:],title="R",label="Hank",titlefont=font(7, "Courier"))
p2 = plot!(IRFRankxi[2,:],title="R",label="Rank",titlefont=font(7, "Courier"))

p3 = plot(IRFxi[ns*na + ns*nx+2,:],title="i",label="Hank",titlefont=font(7, "Courier"))
p3 = plot!(IRFRankxi[11,:],title="i",label="Rank",titlefont=font(7, "Courier"))

p4 = plot(IRFxi[ns*na + ns*nx+3,:],title="M",label="Hank",titlefont=font(7, "Courier"))
p4 = plot!(IRFRankxi[3,:],title="M",label="Rank",titlefont=font(7, "Courier"))

p5 = plot(IRFxi[ns*na + ns*nx+4,:],title="pi",label="Hank",titlefont=font(7, "Courier"))
p5 = plot!(IRFRankxi[4,:],title="pi",label="Rank",titlefont=font(7, "Courier"))

p6 = plot(IRFxi[ns*na + ns*nx+5,:],title="pA",label="Hank",titlefont=font(7, "Courier"))
p6 = plot!(IRFRankxi[5,:],title="pA",label="Rank",titlefont=font(7, "Courier"))

p7 = plot(IRFxi[ns*na + ns*nx+6,:],title="pB",label="Hank",titlefont=font(7, "Courier"))
p7 = plot!(IRFRankxi[6,:],title="pB",label="Rank",titlefont=font(7, "Courier"))

p8 = plot(IRFxi[ns*na + ns*nx+8,:],title="xi",label="Hank",titlefont=font(7, "Courier"))
p8 = plot!(IRFRankxi[8,:],title="xi",label="Rank",titlefont=font(7, "Courier"))

p = plot(p1,p2,p3,p4,p5,p6,p7,p8, layout=(4,2), size=(1600,700))
savefig(p,"xi_irfs.pdf")


p1 = plot(HM0.dGrid,D_ss[1:HM0.nd],xlim=(0.0,5.0),label="unemployed")
p1 = plot!(HM0.dGrid,D_ss[HM0.nd+1:HM0.nd*HM0.ns],xlim=(0.0,5.0),label="employed")
p2 = plot(HM0.aGrid,polC_ss[:,1],label="unemployed")
p2 = plot!(HM0.aGrid,polC_ss[:,2],label="employed")
p3 = plot(polA_ss[:,1],HM0.aGrid,label="unemployed")
p3 = plot!(polA_ss[:,2],HM0.aGrid,label="employed")
p = plot(p1,p2,p3,layout=(1,3),size=(1000,400))
savefig(p,"HankPolicies.pdf")



=#


