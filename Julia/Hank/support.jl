using LinearAlgebra
using Parameters
using IterativeSolvers
using FastGaussQuadrature
using ForwardDiff
using QuantEcon
using Plots
using Arpack

function SolveQZ(Γ0,Γ1,Ψ,Π)
    
    div = 1.0 + 1e-10
    eps = 1e-10
    F = schur!(complex(Γ0),complex(Γ1))
    Lambda, Omega = F.S, F.T
    alpha, beta = F.alpha, F.beta
    Q, Z = adjoint(conj(F.Q)), F.Z

    n = size(Lambda, 1)
    neta = size(Π, 2)

    dLambda = abs.(diag(Lambda))
    dOmega = abs.(diag(Omega))
    dLambda = max.(dLambda,fill(1e-10,size(dLambda))) #to avoid dividing by 0;
    movelast = Bool[(dLambda[i] <= 1e-10) || (dOmega[i] > div * dLambda[i]) for i in 1:n]
    nunstable = sum(movelast)
    nstable = n-nunstable
    iStable = 1:nstable
    iUnstable = (nstable + 1):n

    #Reorder schur to have explosive eigenvalues at the end
    movelastno = fill(false,size(movelast))
    for i in eachindex(movelast)
        movelastno[i] = !movelast[i]
    end
    FS = ordschur!(F, movelastno)
    Lambda, Omega, Q, Z = FS.S, FS.T, FS.Q, FS.Z
    #@show abs.(diag(Lambda))

    gev = hcat(dLambda, dOmega)
    q1 = Q[:,iStable]
    q2 = Q[:,iUnstable]
    q2xΠ = adjoint(q2) * Π
    q2xΨ = adjoint(q2) * Ψ
    q1xΠ = adjoint(q1) * Π
    ndeta1 = min(n - nunstable, neta)
    
    rq2   = rank(q2xΠ)
    rq2q2 = rank([q2xΨ q2xΠ])
    iexist = rq2 == rq2q2
    iunique = rank(Q * Π) == rank(q2xΠ)
    eu = hcat(iexist,iunique)

    #Solve q1xΠ = Phi*q2xΠ by svd decomposition
    #Phi = U1*D1*V1' * V2*inv(D2)*U2
    A2Π = svd(q2xΠ)
    A2Ψ = svd(q2xΨ)
    A1Π = svd(q1xΠ)
    bigevΠ2 = findall(A2Π.S .> eps)
    bigevΨ2 = findall(A2Ψ.S .> eps)
    bigevΠ1 = findall(A1Π.S .> eps)  
    ueta2, deta2, veta2 = A2Π.U[:,bigevΠ2],Matrix(Diagonal(A2Π.S[bigevΠ2])),A2Π.V[:,bigevΠ2]  
    teta, seta, weta = A2Ψ.U[:,bigevΨ2],Matrix(Diagonal(A2Ψ.S[bigevΨ2])),A2Ψ.V[:,bigevΨ2]
    ueta1, deta1, veta1 = A1Π.U[:,bigevΠ1],Matrix(Diagonal(A1Π.S[bigevΠ1])),A1Π.V[:,bigevΠ1]
    Phi =  (ueta1 * deta1 * adjoint(veta1)) * (veta2 * (deta2 \ adjoint(ueta2)))

    #See page 12 of Sims rational expectations document
    L11 = Lambda[iStable,iStable]
    L12 = Lambda[iStable,iUnstable]
    L22 = Lambda[iUnstable,iUnstable]

    O11 = Omega[iStable,iStable]
    O12 = Omega[iStable,iUnstable]
    O22 = Omega[iUnstable,iUnstable]

    Z1 = Z[:,iStable]
    
    #Solve for the effect on lagged variables
    L11inv = LinearAlgebra.inv(L11)
    aux1 = hcat(O11,O12 - Phi*O22) * adjoint(Z)
    aux2 = Z1*LinearAlgebra.inv(L11)
    G1 = real(aux2*aux1)

    #Solve for the effect of exogenous variables (Impact)
    aux3 = vcat(hcat(L11inv, -L11inv*(L12-Phi*L22)),hcat(fill(0.0,(nunstable,nstable)),Matrix(I,nunstable,nunstable)))
    H = Z*aux3
    Impact = real(H * vcat(adjoint(q1) - Phi*adjoint(q2),fill(0.0,(nunstable,size(Ψ,1)))) * Ψ)

    #Solve for the constant 
    #tmat = hcat(Matrix(I,nstable,nstable), -Phi)
    #G0 = vcat(tmat * Lambda, hcat(zeros(nunstable,nstable), Matrix(I,nunstable,nunstable)))
    #G = vcat(tmat * Omega, fill(0.0,(nunstable, n)))
    #G0I = inv(G0)
    #G = G0I * G
    #usix = (nstable + 1):n
    #Ostab = Omega[nstable+1:n,nstable+1:n]
    #Lstab = Lambda[nstable+1:n,nstable+1:n]
    #C = G0I * vcat(tmat * adjoint(Q) * C, (Lstab - Ostab) \ adjoint(q2) * C)

    return eu,G1,Impact
end

function SolveSystem(A,B,C,E,maxit = 1000)
    P0 = fill(0.0,size(A))
    S0 = fill(0.0,size(C))
    for i = 1:maxit
        P = -(A*P0 + B) \ C
        S = -(C*S0 + B) \ A
        @show test = maximum(C + B*P + A*P*P)
        if test<0.0000000001  
            break
        end
        P0 = P
        S0 = S
    end
    Q = -(A*P0 + B)\E
    XP = LinearAlgebra.eigen(P0)
    XS = LinearAlgebra.eigen(S0)
    #@show XP.values
    if maximum(abs.(XP.values)) > 1.0
        error("Non existence")
    end
    if maximum(abs.(XS.values)) > 1.0
        error("No stable equilibrium")
    end

    return P0,Q
end

function TurnABCEtoSims(A,B,C,E)
    HasLead = any((abs.(A) .> 1e-9),dims = 2)
    HasLead = reshape(HasLead,size(A,1))
    Ashift = copy(A)
    Bshift = copy(B)
    Cshift = copy(C)

    Ashift[.!HasLead,:] = B[.!HasLead,:]
    Bshift[.!HasLead,:] = C[.!HasLead,:]
    Cshift[.!HasLead,:] .= 0.0

    #IsLag = findall(any((abs.(Cshift) .> 1e-9),dims=1))
    ##Not sure why I have to use this Linear Indices function, but not using gives me an error in the adjcost case
    IsLag = any((abs.(Cshift) .> 1e-9),dims=1)
    IsLag = LinearIndices(IsLag)[findall(IsLag)]
    n = size(A,1)
    naux = length(IsLag)
    iaux = n+1:n+naux

    G = fill(0.0,(n+naux,n+naux))
    H = fill(0.0,(n+naux,n+naux))

    G[1:n,1:n] = -Ashift
    H[1:n,1:n] = Bshift
    H[1:n,iaux] = Cshift[:,IsLag]

    for (i,col) in enumerate(IsLag)
        G[n+i,n+i] = 1.0
        H[n+i,col] = 1.0
    end

    nEE = length(findall(HasLead))
    EE = fill(0.0,(n+naux,nEE))
    leadeqs = findall(HasLead)
    for (i,lead) in enumerate(leadeqs)
        EE[lead,i] = 1.0
    end
    nE = size(E,2)
    E = vcat(E,fill(0.0,(naux,nE)))
    
    G,H,E,EE = convert(Array{Float64},G),convert(Array{Float64},H),convert(Array{Float64},E),convert(Array{Float64},EE)
    return G,H,E,EE
end
