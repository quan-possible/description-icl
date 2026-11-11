"""Check: does n^2 V_n -> v_inf = E[pi_inf (1-pi_inf) |c(m-w)+eps w|^2] > 0 ?
pi_inf = P(Z=1 | m, w). If so, lim ESS_n = p(c-eps) - v_inf/d (bound not attained)."""
import numpy as np
rng = np.random.default_rng(0)

def draw(d, a0, c, p, B):
    r = 1/(c*a0)
    m = rng.normal(0, np.sqrt((1-r)*a0), (B, d))
    z = rng.random(B) < p
    w = np.where(z[:, None], m + rng.normal(0, np.sqrt(r*a0), (B, d)), rng.normal(0, np.sqrt(a0), (B, d)))
    return m, w

def v_inf(d, a0, c, p, B=2_000_000):
    eps = 1/a0
    m, w = draw(d, a0, c, p, B)
    l1 = np.log(p) + 0.5*d*np.log(c) - 0.5*c*((w-m)**2).sum(1)
    l0 = np.log(1-p) + 0.5*d*np.log(eps) - 0.5*eps*(w**2).sum(1)
    pi = 1/(1+np.exp(np.clip(l0-l1, -700, 700)))
    u = c*(m-w) + eps*w
    val = pi*(1-pi)*(u**2).sum(1)
    return val.mean(), val.std()/np.sqrt(B)

def n2Vn(d, a0, c, p, n, B=20000):
    eps = 1/a0
    m, w = draw(d, a0, c, p, B)
    X = rng.normal(size=(B, n, d)); y = np.einsum('bnd,bd->bn', X, w) + rng.normal(size=(B, n))
    W = np.einsum('bnd,bne->bde', X, X); I = np.eye(d)
    S1 = np.linalg.inv(c*I + W); S0 = np.linalg.inv(eps*I + W)
    Xy = np.einsum('bnd,bn->bd', X, y)
    mu1 = np.einsum('bde,be->bd', S1, c*m + Xy); mu0 = np.einsum('bde,be->bd', S0, Xy)
    e1 = y - np.einsum('bnd,bd->bn', X, m); Xe1 = np.einsum('bnd,bn->bd', X, e1)
    # log marginal likelihoods: N(e; 0, I + k^-1 X X^T)
    def lml(e, Xe, k, S):
        q = (e**2).sum(1) - np.einsum('bd,bde,be->b', Xe, S, Xe)
        return -0.5*np.linalg.slogdet(I + W/k)[1] - 0.5*q
    l1 = np.log(p) + lml(e1, Xe1, c, S1); l0 = np.log(1-p) + lml(y, Xy, eps, S0)
    pi = 1/(1+np.exp(np.clip(l0-l1, -700, 700)))
    val = n*n*pi*(1-pi)*((mu1-mu0)**2).sum(1)   # E over x of ((mu1-mu0)^T x)^2
    return val.mean(), val.std()/np.sqrt(B)

print("d a0 c p | bound p(c-eps) | v_inf/d (se) | exact limit | n^2 V_n/d at n=100, 400")
for d, a0, c, p in [(16,10,100,.9),(16,10,1,.9),(5,10,10,.9),(5,10,1,.9),(5,10,1,.5),(5,10,.5,.9),(2,10,1,.9),(1,10,1,.9),(1,10,1,.5),(1,10,36,.9),(1,10,.36,.9)]:
    v, se = v_inf(d, a0, c, p)
    b = p*(c-1/a0)
    chk = [n2Vn(d, a0, c, p, n, B=20000 if d<=5 else 4000) for n in (100, 400)]
    print(f"{d:2d} {a0} {c:5} {p} | {b:7.3f} | {v/d:8.4f} ({se/d:.4f}) | {b-v/d:7.3f} | " + "  ".join(f"{x/d:.4f}({s/d:.4f})" for x, s in chk))
