import numpy as np
from numpy import kron
from scipy.linalg import expm
from scipy.special import hermite

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
 
def sample_entarr(nmax,nlarge,time,ith_order):
  MM,PM = prepare_hermite_base_MM_PM(nmax,nlarge)
  Udirect = expm(1j * kron(MM,MM) * time)
  single = np.zeros(nmax)
  single[ith_order] = 1
  arr = kron(single,single)
  return Udirect.dot(arr)
 
def entarr_to_rhored(entarr,nmax):
  rho_full = np.reshape(kron(entarr,np.conj(entarr)),(nmax,nmax,nmax,nmax))
  return np.einsum('ijkj->ik',rho_full)

nmax = 10
nlarge = 600
#time parameter should better be smaller than 1, as digitization error will enlarge with time
t_time = 1
beta = 1

sample_entarr = sample_entarr(nmax,nlarge,t_time,1)
sample_rhored = entarr_to_rhored(sample_entarr,nmax)
entropy_exact = EntS_sim(sample_rhored)

MM,PM = prepare_hermite_base_MM_PM(nmax,nlarge)
xvar_exact = abs(np.trace(sample_rhored.dot(MM).dot(MM)))
pvar_exact = abs(np.trace(sample_rhored.dot(PM)))
#gaussian approx value doesn't match with the theory calculation is because the digitization error on var
entropy_gapprox = EntS_Gapprox_2by2diag(xvar_exact,pvar_exact)

#print(xvar_exact)
#print(pvar_exact)
print("exact entropy",entropy_exact)
print("gaussian approximation entropy",entropy_gapprox)

#gmm = 0.03
#gpm = 0.07
#gfourth = 0.087
gmm = 0.1
gpm = 0.1
gfourth = 0.0678

HM_QHO = gmm * MM.dot(MM) + gpm * PM
HM_forth = HM_QHO + gfourth * MM.dot(MM).dot(MM).dot(MM)
density_matrix_forth = rho_can(beta, HM_forth)

#Calculate invariant quantity during approx
energy_exp_therm = abs(np.trace(density_matrix_forth.dot(HM_forth)))
energy_exp_Qstate = abs(np.trace(sample_rhored.dot(HM_forth)))

print("energy_exp_therm",energy_exp_therm)
print("energy_exp_Qstate",energy_exp_Qstate)

entropy_forth = EntS_sim(density_matrix_forth)
print("improved thermal approximation entropy",entropy_forth)
print("\n")
