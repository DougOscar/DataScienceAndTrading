import numpy as np
rng=np.random.default_rng(0)
n=1560*200
for dist in ["gauss","t3"]:
    e = rng.standard_normal(n) if dist=="gauss" else rng.standard_t(3,n)
    x=np.cumsum(e)
    for L in (10,20,60,120):
        D=x[L:]-x[:-L]; s=np.sign(D)
        cross=np.sum(s[1:]*s[:-1]<0)
        rate=cross/len(D)
        # inter-crossing interval stats
        idx=np.where(s[1:]*s[:-1]<0)[0]; gaps=np.diff(idx)
        fr={h:np.mean(gaps>h) for h in (6,12,24,36,60)}
        print(dist,L,"rate/bar %.4f  /yr %.0f  formula %.4f  mean gap %.1f med %.0f  P(gap>hold):"%(rate,rate*1560,np.arccos(1-1/L)/np.pi,gaps.mean(),np.median(gaps)), {k:round(v,2) for k,v in fr.items()})
