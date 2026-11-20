"""Unbiased conditional row estimators; probabilities never use exact psi implicitly."""
import numpy as np

def plan(coefficients, *, top=8, power=1.0, uniform_mix=0.1):
    h=np.asarray(coefficients)
    if h.ndim!=1 or not np.isfinite(h).all():raise ValueError('finite 1D coefficients required')
    if not isinstance(top,int) or top<0 or power<0 or not 0<uniform_mix<=1:
        raise ValueError('invalid proposal parameters')
    active=np.flatnonzero(h!=0)
    count=min(top,active.size)
    selected=active[np.argsort(-np.abs(h[active]),kind='stable')[:count]]
    tail=np.setdiff1d(active,selected)
    q=np.zeros(h.size,dtype=float)
    if tail.size:
        magnitudes=np.abs(h[tail]).astype(np.float64)
        weights=(magnitudes/np.max(magnitudes))**power
        q[tail]=(1-uniform_mix)*weights/weights.sum()+uniform_mix/tail.size
        q[tail]/=q[tail].sum()
    return selected,tail,q

def conditional_variance(contributions,tail,q, *, samples, surrogate=None):
    if not isinstance(samples,int) or samples<1:raise ValueError('positive sample count required')
    residual=np.asarray(contributions,dtype=complex)
    if surrogate is not None:residual=residual-np.asarray(surrogate)
    z=residual[tail]
    if not len(tail):return 0.0
    if np.any(q[tail]<=0) or not np.isclose(q[tail].sum(),1):raise ValueError('invalid tail probabilities')
    return max(0.0,float(np.sum(np.abs(z)**2/q[tail])-np.abs(z.sum())**2))/samples

def estimate(contributions,selected,tail,q, *, samples,rng,surrogate=None):
    if not isinstance(samples,int) or samples<1:raise ValueError('positive sample count required')
    a=np.asarray(contributions,dtype=complex)
    g=np.zeros_like(a) if surrogate is None else np.asarray(surrogate,dtype=complex)
    residual=a-g
    value=g.sum()+residual[selected].sum()
    if tail.size:
        indices=rng.choice(tail,size=samples,replace=True,p=q[tail])
        value+=np.mean(residual[indices]/q[indices])
    return value
