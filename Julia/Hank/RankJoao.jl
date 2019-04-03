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

struct RankParameters{T <: Real}
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
    R_ss::T
end

function Rank(
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
    amin::T = 1e-9,
    amax::T = 200.0,
    aSize::I = 100,
    dSize::I = 100 ) where{T <: Real,I <: Integer}

    #############Params
    R_ss = 1.0/β
    μϵ = μ/(μ-1.0)
    Mbar = (1.0-ubar)*δ/(ubar+δ*(1.0-ubar))
    wbar = 1.0/μ - δ*ψ*Mbar
    ben = wbar*0.5
    params = RankParameters(β,γ,ρz,σz,ρξ,σξ,ζ,ψ,μ,μϵ,θ,ω,ubar,δ,Mbar,wbar,B,ben,amin,R_ss)
    
    return params
end

function uprime(C,params)
    @unpack γ = params
    return C^(-γ)
end


function AggResidual(C,C_P,u,u_L,R,R_P,i_L,i,M,M_P,pi,pi_P,pA,pB,pA_P,pB_P,Y,Z_L,Z,ξ_L,ξ,epsilon::AbstractArray,params::RankParameters)
    @unpack β,γ,ρz,σz,ρξ,σξ,ζ,ψ,μ,μϵ,θ,ω,ubar,δ,Mbar,wbar,B,ben,amin,R_ss = params
    ϵz,ϵξ = epsilon

    #Y = Z*(1.0-u)
    H = 1.0-u - (1.0-δ)*(1.0-u_L)
    marg_cost = (wbar * (M/Mbar)^ζ + ψ*M - (1.0-δ)*ψ*M_P)/Z
    AggEqs = vcat(
        Y - C - ψ*M*H, #bond market clearing
        1.0 + i - R_ss * pi^ω * ξ, #mon pol rule
        R - (1.0 + i_L)/pi, 
        M - (1.0-u-(1.0-δ)*(1.0-u_L))/(u_L + δ*(1.0-u_L)), #3 labor market 
        pi - θ^(1.0/(1.0-μϵ))*(1.0-(1.0-θ)*(pA/pB)^(1.0-μϵ))^(1.0/(μϵ-1.0)), #4 inflation
        -pA + μ*Y*marg_cost + θ*pi_P^μϵ*pA_P/R, #aux inflation equ 1
        -pB + Y + θ*pi_P^(μϵ-1.0)*pB_P/R, #aux inflation equ 2
        log(Z) - ρz*log(Z_L) - σz*ϵz, #TFP evol
        log(ξ) - ρξ*log(ξ_L) - σξ*ϵξ, #Mon shock
        -uprime(C,params) + β * R_P * uprime(C_P,params), #Euler
        -Y + Z*(1-u)
    ) 
    
    return AggEqs
end

function F(X_L,X,X_P,epsilon,params)

    u_L, R_L, M_L, pi_L, pA_L, pB_L, Z_L, ξ_L, C_L, Y_L, i_L = X_L
    u, R, M, pi, pA, pB, Z, ξ, C, Y, i = X
    u_P, R_P, M_P, pi_P, pA_P, pB_P, Z_P, ξ_P, C_P, Y_P, i_P = X_P

    return AggResidual(C,C_P,u,u_L,R,R_P,i_L,i,M,M_P,pi,pi_P,pA,pB,pA_P,pB_P,Y,Z_L,Z,ξ_L,ξ,epsilon,params)
end

#=
RankM = Rank()
@unpack θ,ψ,δ,Mbar,ubar,R_ss = RankM

####Vector of steady state variables
pB  = (1.0-ubar)/(1.0 - θ/R_ss)
C = (1-ubar) - ψ*Mbar * δ * (1.0-ubar)
Agg_SS = [ubar;R_ss;Mbar;1.0;pB;pB;1.0;1.0;C;1-ubar;R_ss-1.0]

####Matrices for rational expectations computation
Amat = ForwardDiff.jacobian(t -> F(Agg_SS,Agg_SS,t,[0.0;0.0],RankM),Agg_SS)
Bmat = ForwardDiff.jacobian(t -> F(Agg_SS,t,Agg_SS,[0.0;0.0],RankM),Agg_SS)
Cmat = ForwardDiff.jacobian(t -> F(t,Agg_SS,Agg_SS,[0.0;0.0],RankM),Agg_SS)
Emat = ForwardDiff.jacobian(t -> F(Agg_SS,Agg_SS,Agg_SS,t,RankM),[0.0;0.0])

@time PP,QQ = SolveSystem(Amat,Bmat,Cmat,Emat)
G,H,E,EE = TurnABCEtoSims(Amat,Bmat,Cmat,Emat)
#@time eu,G1,Impact = SolveQZ(G,H,E,EE)
#@show eu

simul_length = 200
H = Matrix(I,size(PP,1),size(PP,1))
lss = LSS(PP,QQ,H)
X_simul, _ = simulate(lss, simul_length);

Time = 100
IRFRankZ = fill(0.0,(length(Agg_SS),Time))
IRFRankZ[:,1] = QQ[:,1] #z shock
for t =2:Time
    IRFRankZ[:,t] = PP*IRFRankZ[:,t-1]
end

IRFRankxi = fill(0.0,(length(Agg_SS),Time))
IRFRankxi[:,1] = QQ[:,2] #xi shock
for t =2:Time
    IRFRankxi[:,t] = PP*IRFRankxi[:,t-1]
end
=#
