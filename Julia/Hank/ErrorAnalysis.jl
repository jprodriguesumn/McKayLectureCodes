include("HankJoaoCol.jl")
include("HankJoaoEgm.jl")
R0 = 1.01


#########################collocation
HMcol,polcol,Aggscol = Hank(R0)

###fine grid for error analysis
fineGrid = collect(range(HMcol.aGrid[1],stop = HMcol.aGrid[end],length = 10000))
fineGridzero = fill(0.0, HMcol.ns*10000) 

polcol = SolveCollocation(polcol,HMcol,Aggscol,HMcol.params.Penalty)
r,dr = Residual(polcol,HMcol,Aggscol,HMcol.params.Penalty)

colerr = EulerError(Aggscol,Aggscol,polcol,polcol,HMcol,fineGridzero,fineGrid)
colerrm = reshape(colerr,length(fineGrid),HMcol.ns)
@show sum(abs.(colerrm[:,1]))
@show sum(abs.(colerrm[:,2]))
p1 = plot(fineGrid,colerrm[:,1],title="low prod Col error")
p2 = plot(fineGrid,colerrm[:,2],title="high prod Col error")


########################EGM
HM0,pol0,Aggs0 = HankEGM(R0)

cMat = fill(0.0,(HM0.na*HM0.ns,))
aMat = fill(0.0,(HM0.na*HM0.ns,))
transMat = fill(0.0,(HM0.ns*HM0.nd,HM0.ns*HM0.nd))
fineGrid = collect(range(HM0.aGrid[1],stop = HM0.aGrid[end],length = 10000))
fineGridzero = fill(0.0, HM0.ns*10000) 

pol =SolveEGM(pol0,Aggs0,HM0,cMat,aMat)
egmerr = EulerResidualError(pol,pol,Aggs0,Aggs0,HM0,fineGridzero,fineGridzero,vcat(fineGrid,fineGrid))
egmerrm = reshape(egmerr,length(fineGrid),HM0.ns)
@show sum(abs.(egmerrm[:,1]))
@show sum(abs.(egmerrm[:,2]))
p3 = plot(fineGrid,egmerrm[:,1],title="low prod EGM error")
p4 = plot(fineGrid,egmerrm[:,2],title="high prod EGM error")

p = plot(p1,p2,p3,p4,layout=(2,2))
savefig(p,"error.pdf")

