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

mutable struct RBCParameters{T <: Real,I <: Integer}
    β::T
    α::T
    γ::T
    δ::T
    ρ::T
    σz::T #st. deviation of Z shock
    nx::I
end

mutable struct AggVars{T <: Real}
    R::T
    w::T
end

function Prices(K,Z,params::RBCParameters)
    @unpack β,α,δ,γ,ρ,σ,lamw,Lbar = params
    R = Z*α*(K/Lbar)^(α-1.0) + 1.0 - δ
    w = Z*(1.0-α)*(K/Lbar)^(1.0-α) 
    
    return AggVars(R,w)
end

function RBC(
    β = 0.98,
    α = 0.4,
    γ = 2.0,
    δ = 0.02,
    ρ = 0.95,
    σz = 1.0,
    nx = 5)
    return RBCParameters(β,α,γ,δ,ρ,σz,5)
end    

function SteadyState(params)
    @unpack β,α,γ,δ,ρ,σz,nx = params
    Z = 1.0
    R = 1.0/β
    K = ((R - 1.0+δ)/α)^(1.0/(α-1.0))
    Y = K^α
    C = Y - δ*K
    X = [Z;R;K;Y;C]
    return X
end

function F(X_L::AbstractArray,
           X::AbstractArray,
           X_P::AbstractArray,
           epsilon::AbstractArray,
           params::RBCParameters)

    @unpack β,α,γ,δ,ρ,σz,nx = params   
    ϵz = epsilon[1]
    Z_L, R_L, K_L, Y_L, C_L = X_L
    Z, R, K, Y, C = X
    Z_P, R_P, K_P, Y_P, C_P = X_P
    
    eqs = vcat(
        1.0 - β * R_P * C_P^(-γ)*C^(γ),
        R - (α*Z*K_L^(α-1.0) + 1.0 - δ),
        K - (1.0-δ)*K_L + C - Y ,
        Y - Z*K_L^α,
        log(Z) - ρ*log(Z_L) - σz*ϵz
    )
    return eqs
end

params = RBC()
xss = SteadyState(params)
roots = F(xss,xss,xss,[0.0],params)
Amat = ForwardDiff.jacobian(t -> F(xss,xss,t,[0.0],params),xss)
Bmat = ForwardDiff.jacobian(t -> F(xss,t,xss,[0.0],params),xss)
Cmat = ForwardDiff.jacobian(t -> F(t,xss,xss,[0.0],params),xss)
Emat = ForwardDiff.jacobian(t -> F(xss,xss,xss,t,params),[0.0])

Prbc,Qrbc = SolveSystem(Amat,Bmat,Cmat,Emat)
    

simul_length = 200
H = Matrix(I,size(Prbc,1),size(Prbc,1))
lss = LSS(Prbc,Qrbc,H)
X_simul, _ = simulate(lss, simul_length);

Time = 200
IRFrbcZ = fill(0.0,(length(xss),Time))
IRFrbcZ[:,1] = Qrbc[:,1]
for t =2:Time
    IRFrbcZ[:,t] = Prbc*IRFrbcZ[:,t-1]
end

    
