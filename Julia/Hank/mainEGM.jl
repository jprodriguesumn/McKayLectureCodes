include("HankJoaoEgm.jl")
R0 = 1.01
HM,pol,Aggs = HankEGM(R0)

##predefining matrices
cMat = fill(0.0,(HM.na*HM.ns,))
aMat = fill(0.0,(HM.na*HM.ns,))
transMat = fill(0.0,(HM.ns*HM.nd,HM.ns*HM.nd))

polA_ss,polC_ss,D_ss,K_ss,Aggs_ss = equilibriumEGM(pol,HM,R0)

pB = (1.0-HM.params.ubar)/(1.0-HM.params.θ/Aggs_ss.R)
Agg_ss = [HM.params.ubar;Aggs_ss.R;Aggs_ss.R-1.0;HM.params.Mbar;1.0;pB;pB;1.0;1.0]
xss = vcat(reshape(polA_ss,(HM.ns*HM.na,)),D_ss[2:end],Agg_ss)

root = FEGM(xss,xss,xss,[0.0;0.0],HM,Aggs_ss,2)

###Forsome reason, taking derivatives of residual with EGM takes a long time (Not sure why)
Am = ForwardDiff.jacobian(t -> FEGM(xss,xss,t,[0.0;0.0],HM,Aggs_ss,3),xss)
Bm = ForwardDiff.jacobian(t -> FEGM(xss,t,xss,[0.0;0.0],HM,Aggs_ss,2),xss)
Cm = ForwardDiff.jacobian(t -> FEGM(t,xss,xss,[0.0;0.0],HM,Aggs_ss,1),xss)
Em = ForwardDiff.jacobian(t -> FEGM(xss,xss,xss,t,HM,Aggs_ss,1),[0.0;0.0]) 
#end


G,H,E,EE = TurnABCEtoSims(Am,Bm,Cm,Em)
@time eu,G1,Impact = SolveQZ(G,H,E,EE)
#@time P,Q = SolveSystem(Am,Bm,Cm,Em) #### does not work for some reason!!!


###### Figures
Time = 100
IRFZ = fill(0.0,(size(Impact,1),Time))
IRFZ[:,1] = Impact[:,1]
for t =2:Time
    IRFZ[:,t] = G1*IRFZ[:,t-1]
end
IRFxi = fill(0.0,(size(Impact,1),Time))
IRFxi[:,1] = Impact[:,2]
for t =2:Time
    IRFxi[:,t] = G1*IRFxi[:,t-1]
end
IRFxi = IRFxi[end-9:end-1,:] 
IRFZ = IRFZ[end-9:end-1,:] 



include("RankJoao.jl")
RankM = Rank()
#@unpack θ,ψ,δ,Mbar,ubar,R_ss = RankM

####Vector of steady state variables
pB  = (1.0-RankM.ubar)/(1.0 - RankM.θ/RankM.R_ss)
C = (1-RankM.ubar) - RankM.ψ*RankM.Mbar * RankM.δ * (1.0-RankM.ubar)
Agg_SS = [RankM.ubar;RankM.R_ss;RankM.Mbar;1.0;pB;pB;1.0;1.0;C;1-RankM.ubar;RankM.R_ss-1.0]

####Matrices for rational expectations computation
Amat = ForwardDiff.jacobian(t -> F(Agg_SS,Agg_SS,t,[0.0;0.0],RankM),Agg_SS)
Bmat = ForwardDiff.jacobian(t -> F(Agg_SS,t,Agg_SS,[0.0;0.0],RankM),Agg_SS)
Cmat = ForwardDiff.jacobian(t -> F(t,Agg_SS,Agg_SS,[0.0;0.0],RankM),Agg_SS)
Emat = ForwardDiff.jacobian(t -> F(Agg_SS,Agg_SS,Agg_SS,t,RankM),[0.0;0.0])

@time PP,QQ = SolveSystem(Amat,Bmat,Cmat,Emat)
#G,H,E,EE = TurnABCEtoSims(Amat,Bmat,Cmat,Emat)
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

na = HM.na
ns = HM.ns
nx = HM.nd

p1 = plot(IRFZ[1,:],title="u",label="Hank",titlefont=font(7, "Courier"))
p1 = plot!(IRFRankZ[1,:],title="u",label="Rank",titlefont=font(7, "Courier"))

p2 = plot(IRFZ[2,:],title="R",label="Hank",titlefont=font(7, "Courier"))
p2 = plot!(IRFRankZ[2,:],title="R",label="Rank",titlefont=font(7, "Courier"))

p3 = plot(IRFZ[3,:],title="i",label="Hank",titlefont=font(7, "Courier"))
p3 = plot!(IRFRankZ[11,:],title="i",label="Rank",titlefont=font(7, "Courier"))

p4 = plot(IRFZ[4,:],title="M",label="Hank",titlefont=font(7, "Courier"))
p4 = plot!(IRFRankZ[3,:],title="M",label="Rank",titlefont=font(7, "Courier"))

p5 = plot(IRFZ[5,:],title="pi",label="Hank",titlefont=font(7, "Courier"))
p5 = plot!(IRFRankZ[4,:],title="pi",label="Rank",titlefont=font(7, "Courier"))

p6 = plot(IRFZ[6,:],title="pA",label="Hank",titlefont=font(7, "Courier"))
p6 = plot!(IRFRankZ[5,:],title="pA",label="Rank",titlefont=font(7, "Courier"))

p7 = plot(IRFZ[7,:],title="pB",label="Hank",titlefont=font(7, "Courier"))
p7 = plot!(IRFRankZ[6,:],title="pB",label="Rank",titlefont=font(7, "Courier"))

p8 = plot(IRFZ[8,:],title="Z",label="Hank",titlefont=font(7, "Courier"))
p8 = plot!(IRFRankZ[7,:],title="Z",label="Rank",titlefont=font(7, "Courier"))

p = plot(p1,p2,p3,p4,p5,p6,p7,p8, layout=(4,2), size=(1600,700))
savefig(p,"z_irfs.pdf")

p1 = plot(IRFxi[1,:],title="u",label="Hank",titlefont=font(7, "Courier"))
p1 = plot!(IRFRankxi[1,:],title="u",label="Rank",titlefont=font(7, "Courier"))

p2 = plot(IRFxi[2,:],title="R",label="Hank",titlefont=font(7, "Courier"))
p2 = plot!(IRFRankxi[2,:],title="R",label="Rank",titlefont=font(7, "Courier"))

p3 = plot(IRFxi[3,:],title="i",label="Hank",titlefont=font(7, "Courier"))
p3 = plot!(IRFRankxi[11,:],title="i",label="Rank",titlefont=font(7, "Courier"))

p4 = plot(IRFxi[4,:],title="M",label="Hank",titlefont=font(7, "Courier"))
p4 = plot!(IRFRankxi[3,:],title="M",label="Rank",titlefont=font(7, "Courier"))

p5 = plot(IRFxi[5,:],title="pi",label="Hank",titlefont=font(7, "Courier"))
p5 = plot!(IRFRankxi[4,:],title="pi",label="Rank",titlefont=font(7, "Courier"))

p6 = plot(IRFxi[6,:],title="pA",label="Hank",titlefont=font(7, "Courier"))
p6 = plot!(IRFRankxi[5,:],title="pA",label="Rank",titlefont=font(7, "Courier"))

p7 = plot(IRFxi[7,:],title="pB",label="Hank",titlefont=font(7, "Courier"))
p7 = plot!(IRFRankxi[6,:],title="pB",label="Rank",titlefont=font(7, "Courier"))

p8 = plot(IRFxi[9,:],title="xi",label="Hank",titlefont=font(7, "Courier"))
p8 = plot!(IRFRankxi[8,:],title="xi",label="Rank",titlefont=font(7, "Courier"))

p = plot(p1,p2,p3,p4,p5,p6,p7,p8, layout=(4,2), size=(1600,700))
savefig(p,"xi_irfs.pdf")

#=if Collocation 
    p1 = plot(HM.aGrid,polm_ss[:,1], label="unemployed")
    p1 = plot!(HM.aGrid,polm_ss[:,2], label="employed")
    p1 = plot!(HM.aGrid,HM.aGrid, line = :dot, label="45 degree line")
    p2 = plot(HM.aGrid,cpol_ss[:,1], label="unemployed")
    p2 = plot!(HM.aGrid,cpol_ss[:,2], label="employed")
    p3 = plot(HM.dGrid,distm_ss[:,1],label="unemployed")
    p3 = plot!(HM.dGrid,distm_ss[:,2],label="employed")
    p3 = plot!(xlims=(0.0,5.0))
    p = plot(p1,p2,p3, layout=(1,3),size=(1000,400))
    savefig(p,"HankPolicies.pdf")
end=#
