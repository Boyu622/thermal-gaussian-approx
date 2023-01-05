import numpy as np
from numpy import kron
from scipy.linalg import expm
from scipy.special import hermite
from scipy.integrate import quad

#initialize digitization, parameter in H and temperature

gmm = 1
gpm = 1
gfourth = 1.0
beta = 1.0

#define canonical density matrix
def rho_can(beta, HM):
  return expm(-beta*HM) / np.trace(expm(-beta*HM))
  
def prepare_MM_PM(nmax):
  xarr = []
  for i in range(nmax): xarr.append( (i+1)*20/nmax - (10) - 20/(2*nmax) )
  MM = np.diag(xarr)
  PM = (xarr[2] - xarr[1])**(-2)*(np.diag(np.full(nmax, 2)) + np.diag(np.full(nmax-1, -1), 1) + np.diag(np.full(nmax-1, -1), -1) )
  return xarr,MM,PM
  
def EntS_sim(mat):
  w,v = np.linalg.eig(mat)
  val = 0
  for ele in w:
    if ele != 0: val -= ele*np.log(ele)
  return abs(val)
  
def EntS_Gapprox_2by2diag(xvar, pvar):
  val = 4*xvar*pvar
  return (1/2 + np.sqrt(val) / 2)*np.log(1/2 + np.sqrt(val) / 2) - (-1/2 + np.sqrt(val) / 2)*np.log(-1/2 + np.sqrt(val) / 2)
  
#This function takes a xarr and return a n-th order hermite-gaussian yarr.
def hermite_gaussian(n,xarr):
  return_arr = []
  for x in xarr: return_arr.append(  hermite(n)(x)*np.exp(-x**2/2)/np.sqrt( np.sqrt(np.pi)*2**n*np.math.factorial(n) )   )
  return return_arr
  
def prepare_hermite_base_MM_PM(nmax,nlarge):
  xarr,MM,PM = prepare_MM_PM(nmax)
  xarrlarge,MMlarge,PMlarge = prepare_MM_PM(nlarge)
  
  hermite_base_MM = np.zeros([nmax,nmax])
  hermite_base_PM = np.zeros([nmax,nmax])
  
  for i in range(nmax):
    ai = hermite_gaussian(i,xarrlarge)
    for j in range(nmax):
      aj = hermite_gaussian(j,xarrlarge)
      valmm = np.transpose(ai).dot(MMlarge).dot(aj) * (xarrlarge[2] - xarrlarge[1])
      valpm = np.transpose(ai).dot(PMlarge).dot(aj) * (xarrlarge[2] - xarrlarge[1])
      hermite_base_MM[i,j] = valmm
      hermite_base_PM[i,j] = valpm
  return hermite_base_MM, hermite_base_PM
    
MM,PM = prepare_hermite_base_MM_PM(10,600)

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

######################

##TWO COPY CASE##
#get position matrix and momentum square matrix for trace method
idenm = np.diag(np.full(10, 1))
M1 = gfourth*3* kron( MM.dot(MM),MM.dot(MM) )
HM_QHO1 = kron(HM_QHO , idenm)
HM_QHO2 = kron(idenm , HM_QHO)
MM4 = MM.dot(MM).dot(MM).dot(MM)
MM41 = gfourth*0.5*kron(MM4 , idenm)
MM42 = gfourth*0.5*kron(idenm , MM4)

#global and local states
density_matrix_Gglobal = rho_can(beta,HM_QHO1+HM_QHO2+M1+MM41+MM42)
density_matrix_Gglobal_re = np.reshape(density_matrix_Gglobal,(10,10,10,10))
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

######################

##FOUR COPY CASE##
HM_QHO1_4 = kron(kron(kron(HM_QHO , idenm),idenm),idenm)
HM_QHO2_4 = kron(kron(kron(idenm , HM_QHO),idenm),idenm)
HM_QHO3_4 = kron(kron(kron(idenm , idenm),HM_QHO),idenm)
HM_QHO4_4 = kron(kron(kron(idenm , idenm),idenm),HM_QHO)

MM41_4 = gfourth*0.25*kron(kron(kron(MM4 , idenm),idenm),idenm)
MM42_4 = gfourth*0.25*kron(kron(kron(idenm , MM4),idenm),idenm)
MM43_4 = gfourth*0.25*kron(kron(kron(idenm , idenm),MM4),idenm)
MM44_4 = gfourth*0.25*kron(kron(kron(idenm , idenm),idenm),MM4)


M11_4 = gfourth*1.5* kron(kron(kron( MM.dot(MM),MM.dot(MM) ),idenm),idenm)
M12_4 = gfourth*1.5* kron(kron(kron( MM.dot(MM),idenm ),MM.dot(MM)),idenm)
M13_4 = gfourth*1.5* kron(kron(kron( MM.dot(MM),idenm ),idenm),MM.dot(MM))
M14_4 = gfourth*1.5* kron(kron(kron( idenm,MM.dot(MM) ),MM.dot(MM)),idenm)
M15_4 = gfourth*1.5* kron(kron(kron( idenm,MM.dot(MM) ),idenm),MM.dot(MM))
M16_4 = gfourth*1.5* kron(kron(kron( idenm,idenm ),MM.dot(MM)),MM.dot(MM))

Mtot_4 = gfourth*6*kron(kron(kron( MM,MM ),MM),MM)

density_matrix_Gglobal_4 = rho_can(beta,HM_QHO1_4+HM_QHO2_4+HM_QHO3_4+HM_QHO4_4+MM41_4+MM42_4+MM43_4+MM44_4+M11_4+M12_4+M13_4+M14_4+M15_4+M16_4+Mtot_4)
density_matrix_Gglobal_re_4 = np.reshape(density_matrix_Gglobal_4,(10,10,10,10,10,10,10,10))
density_matrix_Glocal_4 = np.einsum('ijlmkjlm->ik',density_matrix_Gglobal_re_4)
#second moment and output
xvariance_Glocal_4 = np.trace(density_matrix_Glocal_4.dot(MM).dot(MM))
pvariance_Glocal_4 = np.trace(density_matrix_Glocal_4.dot(PM))
entropy_Glocal_4 = EntS_sim(density_matrix_Glocal_4)
print("x / p variance and entropy for QHO+x^4 (four copy case)")
print("xvariance",xvariance_Glocal_4)
print("pvariance",pvariance_Glocal_4)
print("entropy",entropy_Glocal_4)
print("\n")
