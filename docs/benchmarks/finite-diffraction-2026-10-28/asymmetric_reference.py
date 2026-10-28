import numpy as np,json
from scipy.integrate import quad
S=np.array([-2.,1.,-8.]);nu=.5;c=343
result=[]
for y in [-1.005,-.995,-2.]:
 R=np.array([2.,y,4.]);rs=np.linalg.norm(S[:2]);rr=np.linalg.norm(R[:2]);ts=np.arctan2(S[0],-S[1])%(2*np.pi);tr=np.arctan2(R[0],-R[1])%(2*np.pi)
 phi=nu*(np.pi+np.array([ts+tr,ts-tr,-ts+tr,-ts-tr]));center=(S[2]*rr+R[2]*rs)/(rs+rr)
 m=np.hypot(rs,S[2]-center);l=np.hypot(rr,R[2]-center);widths=2*np.abs(np.sin(phi/2))*(m*l/(m+l))/nu
 points=sorted(set([-100.,100.,center]+[float(np.clip(center+sgn*w*factor,-100,100)) for w in widths for factor in [.01,.1,1,10,100] for sgn in [-1,1]]))
 # Integrate each smooth interval separately; refine tolerances to check stability.
 fields=[]
 for tol in [1e-8,1e-11]:
  values=[]
  for hz in [0,125,250,500]:
   def fun(z):
    zs=S[2]-z;zr=R[2]-z;m=np.hypot(rs,zs);l=np.hypot(rr,zr);eta=np.arcsinh(zs/rs)+np.arcsinh(zr/rr)
    beta=np.sum(np.sin(phi)/(2*(np.sinh(nu*eta/2)**2+np.sin(phi/2)**2)))
    return (-nu*beta/(4*np.pi*m*l))*np.exp(-2j*np.pi*hz*(m+l)/c) if m+l<=34.3 else 0j
   v=sum(quad(lambda z:fun(z).real,a,b,epsabs=tol,limit=100)[0]+1j*quad(lambda z:fun(z).imag,a,b,epsabs=tol,limit=100)[0] for a,b in zip(points,points[1:]))
   # Direct path crosses x=0 at y=(1+y)/2, above the sheet when y>-1.
   if y>-1:v+=np.exp(-2j*np.pi*hz*np.linalg.norm(R-S)/c)/np.linalg.norm(R-S)
   values.append([v.real,v.imag])
  fields.append(values)
 result.append(dict(y=y,frequencies=[0,125,250,500],values=fields[-1],refinementMax=float(np.max(abs(np.array(fields[0])-fields[1])))))
print(json.dumps(result,indent=2))
