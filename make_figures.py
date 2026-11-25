"""Draw the three figures of the paper.
  python make_figures.py        # writes figures/fig_workflow.png (Fig. 1), fig_tradeoff.png (Fig. 2), fig_trajectory.png (Fig. 3)
Fig. 2 reads results/summary.json (run summarize.py first if you re-ran the study); Figs. 1 and 3 are drawn directly."""
import os, json
os.makedirs('figures', exist_ok=True)
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import cbkt_sim as s
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})

# ---------- trajectory figure
C,R,TL,ty,tr,te=s.generate(42,*s.SCEN['moderate'])
PG,PS,PT,PL0=s.fit_em(C[tr])
std=s.run(C[te],None,PG,PS,PT,PL0); cb=s.run(C[te],R[te]/5.0,PG,PS,PT,PL0)
Ct,Rt,TLt,tyt=C[te],R[te],TL[te],ty[te]
gain=np.abs(std-TLt).mean(1)-np.abs(cb-TLt).mean(1)
mastered=TLt.any(1); m=np.where(mastered,TLt.argmax(1)+1,0)
names=['Calibrated','Overconfident','Underconfident']; picks=[]
for k in range(3):
    ids=np.where((tyt==k)&mastered&(m>=4)&(m<=9))[0]       # learners who mastered mid-session
    med=np.sort(gain[ids])[len(ids)//2]                     # median gain among them
    picks.append(ids[np.argmin(np.abs(gain[ids]-med))])
fig,axs=plt.subplots(1,3,figsize=(7.1,2.3),sharey=True)
x=np.arange(1,s.T_OPP+1)
for ax,k,i in zip(axs,range(3),picks):
    ax.step(x,TLt[i],where='mid',color='#999999',lw=3,alpha=.6,label='True mastery (0/1)')
    ax.plot(x,std[i],'o-',color='#555555',ms=3,lw=1.2,label='stdBKT')
    ax.plot(x,cb[i],'s--',color='black',ms=3,lw=1.2,label='cBKT')
    ax.axhline(.85,color='gray',ls=':',lw=.8)
    for t in range(s.T_OPP):
        ax.text(x[t],-0.13,('✓' if Ct[i,t] else '✗')+'\n'+str(Rt[i,t]),ha='center',va='top',fontsize=6.2,
                color='black' if Ct[i,t] else '#333333',linespacing=.95)
    ax.set_title(f'({"abc"[k]}) {names[k]} learner',fontsize=8.5); ax.set_xlim(.5,15.5); ax.set_ylim(-0.03,1.05)
    ax.set_xticks([1,5,10,15]); ax.grid(alpha=.25)
    ax.text(.6,.865,'τ = 0.85',fontsize=6.5,color='#555555',va='bottom')
    ax.set_xticks([])
axs[0].set_ylabel('Mastery estimate'); axs[1].set_xlabel('Practice opportunity 1–15  (✓/✗ = answer; number = rating 1–5)',labelpad=22)
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=3,fontsize=7,frameon=False,bbox_to_anchor=(.55,1.0))
fig.subplots_adjust(left=.07,right=.995,top=.8,bottom=.3,wspace=.06)
fig.savefig('figures/fig_trajectory.png',dpi=300,bbox_inches='tight',pad_inches=.03)
for k,i in zip(names,picks): print(k,'test idx',i,'mastered at',m[i],'gain MAE',round(gain[i],3))

# ---------- workflow figure (one column)
fig,ax=plt.subplots(figsize=(3.45,3.0)); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
def box(x,y,w,h,t,fc='white',fs=7):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.15',fc=fc,ec='black',lw=.8))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,linespacing=1.15)
def arr(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,lw=.8,color='black',shrinkA=0,shrinkB=0))
RX,RW=4.3,5.6; LX,LW=0.1,3.5; H=1.5
ys=[8.3,6.4,4.5,2.6,0.7]
box(RX,ys[0],RW,H,'Learner rates confidence\n$r_t$ (1–5)',fc='#eeeeee')
box(RX,ys[1],RW,H,'Certainty factor\n$CF_t=r_t/5$')
box(RX,ys[2],RW,H,'$m_t=1.5-CF_t$;  $g_t$ = clip($m_tP_G$)\n$s_t$ = clip($m_tP_S$)',fs=6.6)
box(RX,ys[3],RW,H,'Bayes update of $P(L_t)$,\nthen learning step ($P_T$ fixed)',fs=6.6)
box(RX,ys[4],RW,H,'Mastery estimate ≥ τ ?',fc='#dddddd')
box(LX,ys[2],LW,H,'Fitted once per skill\n(EM, correctness only):\n$P_G, P_S, P_T, P_{L0}$',fc='#f4f4f4',fs=6.2)
box(LX,ys[3],LW,H,'Learner answers\ncorrect $c_t$',fc='#eeeeee')
for a,b in zip(ys[:-1],ys[1:]): arr(RX+RW/2,a,RX+RW/2,b+H)
arr(LX+LW,ys[2]+H/2,RX,ys[2]+H/2)
arr(LX+LW,ys[3]+H/2,RX,ys[3]+H/2)
ax.text(LX+LW/2,ys[4]+H/2,'yes: advance\nno: next item',ha='center',va='center',fontsize=6.4)
fig.subplots_adjust(0,0,1,1); fig.savefig('figures/fig_workflow.png',dpi=300,bbox_inches='tight',pad_inches=.03)

# ---------- trade-off curves (Fig. 2)
plt.rcParams.update({'font.family': plt.rcParamsDefault['font.family'], 'font.size': plt.rcParamsDefault['font.size']})
S = json.load(open('results/summary.json'))
fig, ax = plt.subplots(1, 2, figsize=(16.285, 6.68), dpi=200, sharey=True)
for a_, (k, t) in zip(ax, (('moderate', '(a) Moderate miscalibration'), ('severe', '(b) Severe miscalibration'))):
    X = S[k]
    for m, lab, st, mk, c in (('std', 'stdBKT', '-', 'o', '#555555'), ('cb', 'cBKT', '--', 's', 'black')):
        a_.plot(X[m]['over'], 100 * np.array(X[m]['under']), st, marker=mk, ms=5, lw=2.4, color=c, label=lab)
    a_.axhline(5, color='gray', ls=':', lw=1.3); a_.set_title(t, fontsize=17)
    a_.set_xlim(0, 6); a_.set_ylim(0, 45); a_.grid(alpha=.3); a_.tick_params(labelsize=14)
    a_.set_xlabel('Over-practice\n(mean unnecessary items)', fontsize=16)
ax[0].set_ylabel('Under-practice rate\n(% learners advanced early)', fontsize=16)
ax[0].text(4.4, 5.8, '5% tolerance', color='#555', fontsize=13); ax[0].legend(fontsize=15, loc='upper right')
plt.tight_layout(); plt.savefig('figures/fig_tradeoff.png')
print('wrote figures/fig_workflow.png, fig_tradeoff.png, fig_trajectory.png')
