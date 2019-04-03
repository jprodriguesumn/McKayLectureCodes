include("HankJoaoCol.jl")
R0 = 1.01
HM,pol,Aggs = Hank(R0)

pol_ss,polm_ss,EA_ss,R_ss,dist_ss,distm_ss,cpol_ss = equilibrium(pol,HM,1.01)


#######################Aggregates at steady state
pB = (1.0-HM.params.ubar)/(1.0-HM.params.θ/R_ss)
Agg_ss = [HM.params.ubar;R_ss;R_ss-1.0;HM.params.Mbar;1.0;pB;pB;1.0;1.0]
xss = vcat(reshape(pol_ss,HM.ns*HM.na),dist_ss[2:end],Agg_ss)    

@show maximum(F(xss,xss,xss,[0.0;0.0],HM,1))

@time Am = ForwardDiff.jacobian(t -> F(xss,xss,t,[0.0;0.0],HM,3),xss)
Bm = ForwardDiff.jacobian(t -> F(xss,t,xss,[0.0;0.0],HM,2),xss)
Cm = ForwardDiff.jacobian(t -> F(t,xss,xss,[0.0;0.0],HM,1),xss)
Em = ForwardDiff.jacobian(t -> F(xss,xss,xss,t,HM,1),[0.0;0.0])

G,H,E,EE = TurnABCEtoSims(Am,Bm,Cm,Em)
@time eu,G1,Impact = SolveQZ(G,H,E,EE)
#@time P,Q = SolveSystem(Am,Bm,Cm,Em) #### does not work for some reason!!!

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

#=
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
=#
