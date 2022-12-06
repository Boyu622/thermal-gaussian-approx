import numpy as np
from scipy.linalg import expm
from numpy import kron

#initialize digitization, parameter in H and temperature
nmax_direct = 128
nmax_trace2 = 64
nmax_direct_six = 128
nmax_trace2_six = 64
gmm = 0.3
gpm = 0.5
gfourth = 0.7
gsixth = 0.9
beta = 2.0

#define canonical density matrix
def rho_can(beta, HM):
  return expm(-beta*HM) / np.trace(expm(-beta*HM))
def prepare_MM_PM(nmax):
  xarr = []
  for i in range(nmax): xarr.append( (i+1)*8/nmax - (4) - 8/(2*nmax) )
  MM = np.diag(xarr)
  PM = (xarr[2] - xarr[1])**(-2)*(np.diag(np.full(nmax, 2)) + np.diag(np.full(nmax-1, -1), 1) + np.diag(np.full(nmax-1, -1), -1) )
  return MM,PM
def EntS_sim(mat):
  w,v = np.linalg.eig(mat)
  val = 0
  for ele in w:
    if ele != 0: val -= ele*np.log(ele)
  return abs(val)
def EntS_Gapprox_2by2diag(xvar, pvar):
  val = 4*xvar*pvar
  return (1/2 + np.sqrt(val) / 2)*np.log(1/2 + np.sqrt(val) / 2) - (-1/2 + np.sqrt(val) / 2)*np.log(-1/2 + np.sqrt(val) / 2)

print("\n")
print("1. these cases are supposed to have same covariance matrix by Wolf's paper")
print("2. with increasing of copies, the state tends to gaussian")
print("3. the higher the n is, the more similar this state is with respect to gaussian")
print("\n")

##DIRECT SIMULATION for QHO+x^4##
#get position matrix and momentum square matrix for direct method
MM,PM = prepare_MM_PM(nmax_direct)
#our forth order Hamiltonian and corresponding thermal state
HM_QHO = gmm * MM.dot(MM) + gpm * PM
HM_forth = HM_QHO + gfourth * MM.dot(MM).dot(MM).dot(MM)
density_matrix_forth = rho_can(beta, HM_forth)
#second moment and output
xvariance_forth = np.trace(density_matrix_forth.dot(MM).dot(MM))
pvariance_forth = np.trace(density_matrix_forth.dot(PM))
entropy_forth = EntS_sim(density_matrix_forth)
print("x / p variance and entropy for QHO+x^4 (direct simulation)")
print("xvariance",xvariance_forth)
print("pvariance",pvariance_forth)
print("entropy",entropy_forth)
#value from gaussian approximation
entropy_Gapprox = EntS_Gapprox_2by2diag(xvariance_forth,pvariance_forth)
print("Entropy from Gaussian approximation: ", entropy_Gapprox)
print("\n")


##TWO COPY CASE##
#get position matrix and momentum square matrix for trace method
MM,PM = prepare_MM_PM(nmax_trace2)
HM_QHO = gmm * MM.dot(MM) + gpm * PM
idenm = np.diag(np.full(nmax_trace2, 1))

M1 = gfourth*3* kron( MM.dot(MM),MM.dot(MM) )
HM_QHO1 = kron(HM_QHO , idenm)
HM_QHO2 = kron(idenm , HM_QHO)
MM4 = MM.dot(MM).dot(MM).dot(MM)
MM41 = gfourth*0.5*kron(MM4 , idenm)
MM42 = gfourth*0.5*kron(idenm , MM4)

#global and local states
density_matrix_Gglobal = rho_can(beta,HM_QHO1+HM_QHO2+M1+MM41+MM42)
density_matrix_Gglobal_re = np.reshape(density_matrix_Gglobal,(nmax_trace2,nmax_trace2,nmax_trace2,nmax_trace2))
density_matrix_Glocal = np.einsum('ijkj->ik',density_matrix_Gglobal_re)
#second moment and output
xvariance_Glocal = np.trace(density_matrix_Glocal.dot(MM).dot(MM))
pvariance_Glocal = np.trace(density_matrix_Glocal.dot(PM))
entropy_Glocal = EntS_sim(density_matrix_Glocal)
print("x / p variance and entropy for QHO+x^4 (two copy case)")
print("xvariance",xvariance_Glocal)
print("pvariance",pvariance_Glocal)
print("entropy",entropy_Glocal)
print("\n")

##DIRECT SIMULATION QHO+x^4+x^6##
#get position matrix and momentum square matrix for direct method
MM,PM = prepare_MM_PM(nmax_direct_six)
#our forth order Hamiltonian and corresponding thermal state
HM_QHO = gmm * MM.dot(MM) + gpm * PM
HM_sixth = HM_QHO + gfourth * MM.dot(MM).dot(MM).dot(MM) + gsixth * MM.dot(MM).dot(MM).dot(MM).dot(MM).dot(MM)
density_matrix_sixth = rho_can(beta, HM_sixth)
#second moment and output
xvariance_sixth = np.trace(density_matrix_sixth.dot(MM).dot(MM))
pvariance_sixth = np.trace(density_matrix_sixth.dot(PM))
entropy_sixth = EntS_sim(density_matrix_sixth)
print("x / p variance and entropy for QHO+x^4+x^6 (direct simulation)")
print("xvariance",xvariance_sixth)
print("pvariance",pvariance_sixth)
print("entropy",entropy_sixth)
#value from gaussian approximation
entropy_Gapprox = EntS_Gapprox_2by2diag(xvariance_sixth,pvariance_sixth)
print("Entropy from Gaussian approximation: ", entropy_Gapprox)
print("\n")

##TWO COPY CASE QHO+x^4+x^6##
#get position matrix and momentum square matrix for trace method
MM,PM = prepare_MM_PM(nmax_trace2_six)
HM_QHO = gmm * MM.dot(MM) + gpm * PM
idenm = np.diag(np.full(nmax_trace2_six, 1))

M1 = gfourth*3* kron( MM.dot(MM),MM.dot(MM) )
HM_QHO1 = kron(HM_QHO , idenm)
HM_QHO2 = kron(idenm , HM_QHO)
MM4 = MM.dot(MM).dot(MM).dot(MM)
MM41 = gfourth*0.5*kron(MM4 , idenm)
MM42 = gfourth*0.5*kron(idenm , MM4)

M61 = gsixth*3.75* ( kron( MM.dot(MM).dot(MM).dot(MM),MM.dot(MM) ) + kron( MM.dot(MM),MM.dot(MM).dot(MM).dot(MM) ) )
M62 = gsixth*0.25* ( kron( MM.dot(MM).dot(MM).dot(MM).dot(MM).dot(MM), idenm) + kron( idenm, MM.dot(MM).dot(MM).dot(MM).dot(MM).dot(MM)) )

#global and local states
density_matrix_Gglobal = rho_can(beta,HM_QHO1+HM_QHO2+M1+MM41+MM42+M61+M62)
density_matrix_Gglobal_re = np.reshape(density_matrix_Gglobal,(nmax_trace2,nmax_trace2,nmax_trace2,nmax_trace2))
density_matrix_Glocal = np.einsum('ijkj->ik',density_matrix_Gglobal_re)
#second moment and output
xvariance_Glocal = np.trace(density_matrix_Glocal.dot(MM).dot(MM))
pvariance_Glocal = np.trace(density_matrix_Glocal.dot(PM))
entropy_Glocal = EntS_sim(density_matrix_Glocal)
print("x / p variance and entropy for QHO+x^4+x^6 (two copy case)")
print("xvariance",xvariance_Glocal)
print("pvariance",pvariance_Glocal)
print("entropy",entropy_Glocal)
print("\n")
