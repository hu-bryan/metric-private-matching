"""Public block geometry, report postprocessing, and budgeted private adaptation.

Squared Euclidean cost throughout. Every decoder retains every public slot.
OAS is an empirical-prior heuristic, not a worst-case utility theorem.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax
from private_matching.integral_mechanisms import Geometry, solve
from private_matching.mechanisms import sample_euclidean_laplace

BASE_ARMS=("gumbel","span_radial","coordinate_allocated","block_radial",
           "span_oas","coordinate_oas","block_oas","block_bounded_oas","ambient_radial")
ARMS=BASE_ARMS+("public_surrogate_choice","portfolio_40_40_20",
              "portfolio_30_30_40","guard_gumbel","guard_block_oas",
              "guard_public_choice","robust_public_choice","guard_robust_choice",
              "cert_gumbel","cert_block_oas","cert_public_choice","cert_robust_choice")


@dataclass
class PublicPlan:
    blocks:list
    rates:np.ndarray
    coordinate_rates:np.ndarray
    prior_variance:np.ndarray
    oas_shrinkage:float
    coefficient:float
    block_noise_variance:np.ndarray


def public_plan(geo):
    n,d=geo.Y.shape;r=geo.rank
    if r<1:raise ValueError("positive public rank required")
    # Exact public interval diameters from all pairs; no subset of matchings.
    diff=geo.V[:,None,:]-geo.V[None,:,:]
    prefix=np.concatenate((np.zeros((n,n,1)),np.cumsum(diff*diff,axis=2)),axis=2)
    costs=np.full((r+1,r+1),np.inf)
    for end in range(1,r+1):
        for start in range(end):
            k=end-start;D=float(np.sqrt(np.max(prefix[:,:,end]-prefix[:,:,start])))
            A=D*min(k,2*np.sqrt(k*np.log(n)))
            costs[start,end]=A**(2/3)
    best=np.full(r+1,np.inf);prev=np.zeros(r+1,int);best[0]=0.
    for end in range(1,r+1):
        prev[end]=np.argmin(best[:end]+costs[:end,end])
        best[end]=best[prev[end]]+costs[prev[end],end]
    blocks=[];end=r
    while end:
        start=int(prev[end]);blocks.append((start,end));end=start
    blocks.reverse()
    weights=np.array([np.sqrt(costs[a,b]) for a,b in blocks])/np.sqrt(best[r])
    assert np.isclose(weights@weights,1.)
    coordinate_rates=geo.widths**(1/3)/np.sqrt(np.sum(geo.widths**(2/3)))
    # Standard scikit-learn OAS variant, computed spectrally in ambient d.
    eig=geo.singular[:r]**2/n
    mu=float(eig.sum()/d);alpha=float(np.sum(eig*eig)/(d*d))
    denominator=(n+1)*(alpha-mu*mu/d)
    rho=min((alpha+mu*mu)/denominator,1.) if denominator>0 else 1.
    prior=(1-rho)*eig+rho*mu
    variance=np.empty(r)
    for (a,b),rate in zip(blocks,weights):variance[a:b]=(b-a+1)/rate**2
    coefficient=float(best[r]**1.5)
    # The menu contains full-span radial and coordinate Laplace endpoints.
    radial=geo.diameter*min(r,2*np.sqrt(r*np.log(n)))
    coord=float(np.sum(geo.widths**(2/3))**1.5)
    assert coefficient<=min(radial,coord)+1e-8*max(1,radial,coord)
    return PublicPlan(blocks,weights,coordinate_rates,prior,float(rho),coefficient,variance)


def unit_noise(geo,plan,rng_for):
    n,d=geo.Y.shape;r=geo.rank
    span=sample_euclidean_laplace(n,r,1.,rng_for("span"))[0]
    ambient=sample_euclidean_laplace(n,d,1.,rng_for("ambient"))[0]@geo.B
    coord=rng_for("coordinate").laplace(size=(n,r))/plan.coordinate_rates
    if plan.blocks==[(0,r)]:block=span.copy();alias="span_radial"
    elif all(b-a==1 for a,b in plan.blocks):block=coord.copy();alias="coordinate_allocated"
    else:
        block=np.empty((n,r));alias=""
        for ib,((a,b),rate) in enumerate(zip(plan.blocks,plan.rates)):
            block[:,a:b]=sample_euclidean_laplace(n,b-a,rate,rng_for(f"block:{ib}"))[0]
    return dict(span=span,coordinate=coord,block=block,ambient=ambient,
                gumbel=rng_for("gumbel").gumbel(size=(n,n)),block_alias=alias)


def report_scores(geo,plan,noise,epsilon,kind):
    if kind=="ambient_radial":
        z=geo.U+noise["ambient"]/epsilon
        var=np.full(geo.rank,(geo.Y.shape[1]+1)/epsilon**2)
    elif kind.startswith("span"):
        z=geo.U+noise["span"]/epsilon
        var=np.full(geo.rank,(geo.rank+1)/epsilon**2)
    elif kind.startswith("coordinate"):
        z=geo.U+noise["coordinate"]/epsilon
        var=2/(epsilon*plan.coordinate_rates)**2
    else:
        z=geo.U+noise["block"]/epsilon
        var=plan.block_noise_variance/epsilon**2
    if "oas" in kind:
        shrink=plan.prior_variance/(plan.prior_variance+var)
        if "bounded" in kind:
            z=z-np.clip((1-shrink)*z,-np.sqrt(var),np.sqrt(var))
        else:z=z*shrink
    return -2*z@geo.V.T


def candidate(geo,plan,noise,epsilon,kind):
    scores=(geo.base2-2*geo.diameter*noise["gumbel"]/epsilon) if kind=="gumbel" else report_scores(geo,plan,noise,epsilon,kind)
    pi=solve(scores)
    return pi,scores


def total_cost(geo,pi):return float(geo.C2[np.arange(len(pi)),pi].sum())


def portfolio(geo,plan,noise,epsilon,fractions,rng):
    """Two private sources, several free decoders, then private selection.

    eps_g+eps_report+eps_selection=eps. RAND is an action whose score is
    its exact expected total cost, followed by an independent permutation.
    """
    fg,fr,fs=fractions
    assert np.isclose(fg+fr+fs,1.)
    candidates=[candidate(geo,plan,noise,fg*epsilon,"gumbel")[0]]
    labels=["gumbel"]
    for kind in ("block_radial","block_oas","block_bounded_oas"):
        pi=candidate(geo,plan,noise,fr*epsilon,kind)[0]
        if not any(np.array_equal(pi,old) for old in candidates):
            candidates.append(pi);labels.append(kind)
    values=[total_cost(geo,pi) for pi in candidates]+[float(geo.C2.sum()/len(geo.Y))]
    temperature=2*geo.diameter/(fs*epsilon)
    probabilities=softmax(-(np.asarray(values)-min(values))/temperature)
    selected=int(rng.choice(len(values),p=probabilities))
    if selected==len(candidates):pi=rng.permutation(len(geo.Y));label="RAND"
    else:pi=candidates[selected];label=labels[selected]
    return pi,dict(selected=label,candidate_count=len(values),conditional_expected_cost=float(probabilities@values),selection_excess_bound=float(temperature*np.log(len(values))),guard_claim=0.)


def guard(geo,pi,epsilon_validation,rng,alpha=.05):
    """Private lower confidence bound on improvement over expected RAND cost.

    The proposal must already be private at eps-eps_validation. Conditioned
    on that proposal, improvement is 2R_Y-Lipschitz in the Euclidean sum metric,
    with R_Y=max_j ||y_j-public_centroid|| (sharper than the diameter bound).
    If rejected, use an independent uniform permutation. For any proposal,
    expected cost <= expected RAND cost + alpha*(2R_Y/eps_validation)/e.
    """
    random_cost=float(geo.C2.sum()/len(geo.Y))
    improvement=random_cost-total_cost(geo,pi)
    scale=2*geo.radius/epsilon_validation
    lower=improvement+rng.laplace(scale=scale)-scale*np.log(1/(2*alpha))
    accept=bool(lower>0)
    result=pi if accept else rng.permutation(len(geo.Y))
    return result,dict(selected="proposal" if accept else "RAND",guard_claim=max(0.,float(lower)),guard_accepted=accept,guard_false_claim=bool(accept and improvement<lower),guard_harmful_accept=bool(accept and improvement<0),guard_harm_bound=float(alpha*scale/np.e),guard_proposal_improvement=float(improvement))


def bound_for(geo,plan,epsilon,kind):
    n=len(geo.Y);r=geo.rank
    if kind=="gumbel":return 2*n*geo.diameter*np.log(n)/epsilon
    if kind=="span_radial":return 2*n*geo.diameter*min(r,2*np.sqrt(r*np.log(n)))/epsilon
    if kind=="coordinate_allocated":return 2*n*np.sum(geo.widths**(2/3))**1.5/epsilon
    if kind=="block_radial":return 2*n*plan.coefficient/epsilon
    if kind=="ambient_radial":
        d=geo.Y.shape[1]
        return 2*n*geo.diameter*min(d,2*np.sqrt(d*np.log(n)))/epsilon
    if kind=="block_bounded_oas":return 2*n*(plan.coefficient+np.sum(geo.widths*np.sqrt(plan.block_noise_variance)))/epsilon
    return None


def robust_choice(values):
    """Public surrogate selection restricted to finite uniform-utility bounds."""
    eligible=("gumbel","span_radial","coordinate_allocated","block_radial","block_bounded_oas","ambient_radial")
    return min(eligible,key=lambda arm:values[BASE_ARMS.index(arm)])


def regret_certificate(geo,pi,optimum,epsilon_validation,rng,alpha=.05):
    """Private upper confidence bound on actual TOTAL regret, not expected loss.

    Condition on a proposal generated at eps-eps_validation. Its regret is
    2D_Y-Lipschitz even on an unbounded private domain; the common quadratic
    row potential cancels before comparing the chosen and optimal matchings.
    """
    regret=total_cost(geo,pi)-optimum
    scale=2*geo.diameter/epsilon_validation
    upper=max(0.,float(regret+rng.laplace(scale=scale)+scale*np.log(1/(2*alpha))))
    return dict(cert_upper_regret=upper,cert_covered=bool(regret<=upper),
                cert_below_rand=bool(upper<geo.C2.sum()/len(geo.Y)-optimum),
                cert_noise_scale=float(scale))


def public_surrogate_choice(geo,plan,epsilon,rng_for,K=6):
    """Learn a choice entirely from public Y; deliberately check misspecification.

    Surrogate private points are Gaussian draws from the public OAS covariance
    in the public span. Each arm uses the same surrogate geometry and fresh
    channel-specific noise. No true private point, cost, or label is inspected.
    """
    n=len(geo.Y);r=geo.rank
    proxy=np.sqrt(plan.prior_variance)*rng_for("proxy_X").normal(size=(n,r))
    proxy_geo=Geometry.build(proxy@geo.B.T+geo.center,geo.Y)
    # Keep the SAME public coordinate basis, not a second PCA convention.
    proxy_geo.B=geo.B;proxy_geo.V=geo.V;proxy_geo.U=proxy
    proxy_geo.base2=-2*proxy@geo.V.T
    values=np.zeros(len(BASE_ARMS))
    for k in range(K):
        noise=unit_noise(proxy_geo,plan,lambda channel:rng_for(f"proxy:{k}:{channel}"))
        for ia,arm in enumerate(BASE_ARMS):
            pi,_=candidate(proxy_geo,plan,noise,epsilon,arm)
            values[ia]+=total_cost(proxy_geo,pi)/K
    return BASE_ARMS[int(values.argmin())],values
