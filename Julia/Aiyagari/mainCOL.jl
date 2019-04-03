include("Aiyagari.jl")
K0 = 48.0 

AA,Guess,Aggs,finel = Aiyagari(48.0)
tmat = fill(0.0,(AA.ns*AA.nd,AA.ns*AA.nd))
pol_ss,polm_ss,K_ss,R_ss,D_ss,Dm_ss,cpol_ss,Aggs_ss = equilibrium(AA,48.0,Guess,tmat,finel)

####error analysis
Fn = 10000 ###number of points on error grid
fineGrid = collect(range(AA.aGrid[1],stop = AA.aGrid[end],length = Fn))
fineGridzero = fill(0.0, AA.ns*Fn) 
ress,apol = EulerResErr(Aggs_ss,pol_ss,pol_ss,AA,fineGridzero,fineGrid)
resm = reshape(ress,Fn,AA.ns)
polm = reshape(apol,Fn,AA.ns)
@show sum(abs.(resm[:,1]))
@show sum(abs.(resm[:,2]))
p1 = plot(fineGrid,log10.(abs.(resm[:,1])),title="low prod EGM error")
p2 = plot(fineGrid,log10.(abs.(resm[:,2])),title="high prod EGM error")
p = plot(p1,p2,layout = (1,2),legend=false)
savefig(p,"ErrorCollocation.pdf")

#####Plot policies
#=p1 = plot(AA.aGrid,polm_ss[:,1], label="unemployed")
p1 = plot!(AA.aGrid,polm_ss[:,2], label="employed")
p1 = plot!(AA.aGrid,AA.aGrid, line = :dot, label="45 degree line")
p2 = plot(AA.aGrid,cpol_ss[:,1], label="unemployed")
p2 = plot!(AA.aGrid,cpol_ss[:,2], label="employed")
p3 = plot(AA.dGrid,Dm_ss[:,1],label="unemployed")
p3 = plot!(AA.dGrid,Dm_ss[:,2],label="employed")
p3 = plot!(xlims=(0.0,AA.dGrid[end]))
p = plot(p1,p2,p3, layout=(1,3),size=(1000,400))
savefig(p,"AiyagariPolicies.pdf")=#

xss = vcat(pol_ss,D_ss[2:end],[K_ss;1.0])
roots = F(xss,xss,xss,[0.0],AA,1)
eps = [0.0]
Am = ForwardDiff.jacobian(t -> F(xss,xss,t,eps,AA,3),xss)
Bm = ForwardDiff.jacobian(t -> F(xss,t,xss,eps,AA,2),xss)
Cm = ForwardDiff.jacobian(t -> F(t,xss,xss,eps,AA,1),xss)
Em = ForwardDiff.jacobian(t -> F(xss,xss,xss,t,AA,1),eps)

###both SIMs and Rendhal's algorithm working. Sims' faster
#PP,QQ = SolveSystem(Am,Bm,Cm,Em)
G,H,E,EE = TurnABCEtoSims(Am,Bm,Cm,Em)
@time eu,G1,Impact = SolveQZ(G,H,E,EE)

Time = 200
IRFaZ = fill(0.0,(size(G1,1),Time))
IRFaZ[:,1] = Impact[:,1]
for t =2:Time
    IRFaZ[:,t] = G1*IRFaZ[:,t-1]
end

include("RBC.jl")

p1 = plot(IRFaZ[end-1,:],title="K",label="Aiyagari",titlefont=font(7, "Courier"))
p1 = plot!(IRFrbcZ[3,:],title="K",label="RBC",titlefont=font(7, "Courier"))
p2 = plot(IRFaZ[end,:],title="Z",label="Aiyagari",titlefont=font(7, "Courier"))
p2 = plot!(IRFrbcZ[1,:],title="Z",label="RBC",titlefont=font(7, "Courier"))
p = plot(p1,p2, layout=(1,2))
savefig(p,"irfCOL.pdf")



















