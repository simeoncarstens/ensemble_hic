import numpy as np
import os
import sys
from ensemble_hic.analysis_functions import load_sr_samples
sys.path.append(os.path.expanduser('~/projects/ensemble_hic/data/eser2017/'))
from yeastlib import CGRep, centromeres, map_chr_pos_to_bead, rDNA_to_left

scale_factor = 56.323 ## in nm

if True:
    s = load_sr_samples('/scratch/scarste/ensemble_hic/eser2017/whole_genome_rDNA23_arbona2017_it3_100structures_sn_725replicas/samples/', 725, 1150, 25, 500, 1)
    X = np.array([x.variables['structures'].reshape(100,-1,3) for x in s])
if False:
    s = load_sr_samples('/scratch/scarste/ensemble_hic/eser2017/whole_genome_rDNA23_arbona2017_it5_50structures_sn_629replicas/samples/', 629, 2750, 25, 1500, 1)
    X = np.array([x.variables['structures'].reshape(100,-1,3) for x in s])
if False:
    s = load_sr_samples('/scratch/scarste/ensemble_hic/eser2017/whole_genome_rDNA150_prior2_arbona2017_fixed_1structures_s_100replicas/samples/', 100, 23300, 100, 15000, 1)
    X = np.array([x.variables['structures'].reshape(-1,3) for x in s])

X = X.reshape(-1,1239,3) * scale_factor
from csb.bio.utils import fit
ncbeads = np.arange(776,799)
ref_struct = X[-1,ncbeads].copy()
ref_struct-= ref_struct.mean(0)
for i in range(len(X)):
    R,t = fit(ref_struct, X[i, ncbeads])
    X[i] = np.dot(X[i], R.T) + t
    
    
def get_rotation_matrix(i_v, unit=None):
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unit is None:
        unit = [1.0, 0.0, 0.0]
    # Normalize vector length
    i_v /= np.linalg.norm(i_v)

    # Get axis
    uvw = np.cross(i_v, unit)

    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)

    #normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    return (
        rcos * np.eye(3) +
        rsin * np.array([
            [ 0, -w,  v],
            [ w,  0, -u],
            [-v,  u,  0]
        ]) +
        (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    )

complete_structure = X.copy()
rDNA_locus = complete_structure[:,776:799].copy()

axes = (complete_structure.mean(1)-rDNA_locus.mean(1)).T
axes/= np.linalg.norm(axes,axis=0)
axes = axes.T

for i in range(len(complete_structure)):

    R = get_rotation_matrix(axes[i])
    t = complete_structure[i].mean(0)
    complete_structure[i,...] = np.dot(complete_structure[i]-t,R.T) + t
    rDNA_locus[i,...] = np.dot(rDNA_locus[i]-t,R.T) + t
    
rDNA_locus_mean = complete_structure[:,776:799,:].mean(0).mean(0)
rotmat = get_rotation_matrix(rDNA_locus_mean)

# X = np.array([np.dot(Y-0*Y.mean(0), rotmat.T) + 0*Y.mean(0) for Y in X])
# X -= X.mean(1)[:,None,:]

rep = CGRep()
cm_beads = rep.calculate_centromere_beads()[:,0]
tm_beads = rep.calculate_telomere_beads()

axes = {'xy': (0, 1),
        'xz': (0, 2),
        'yz': (1, 2)}
sel_axes = 'xy'
sel_beads = cm_beads
# sel_beads = tm_beads
all_beads = np.arange(1239)#np.random.choice(1239, 20)

if False:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    n_bins = 50
    # ax1.hist2d(X[:,sel_beads,axes[0]].ravel(),
    #            X[:,sel_beads,axes[1]].ravel(),
    #            bins=np.linspace(-18,18,n_bins))
    ax1.hist2d(complete_structure[:,sel_beads,axes[sel_axes][0]].ravel(),
               complete_structure[:,sel_beads,axes[sel_axes][1]].ravel(),
               bins=np.linspace(-18,18,n_bins))
    ax1.set_title('centromere beads')
    ax1.scatter(0.,0.,color='r',s=200,alpha=0.1)
    ax1.set_xlabel('{} coordinate [nm]'.format(sel_axes[0]))
    ax1.set_ylabel('{} coordinate [nm]'.format(sel_axes[1]))
    ax2.hist2d(rDNA_locus[:,:,axes[sel_axes][0]].ravel(),
               rDNA_locus[:,:,axes[sel_axes][1]].ravel(),
               bins=np.linspace(-18,18,n_bins))
    ax2.set_title('all beads')
    ax1.set_xlabel('{} coordinate [nm]'.format(sel_axes[0]))
    for ax in (ax1, ax2):
        ax.set_aspect('equal')
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
    #plt.axes().set_aspect('equal', 'datalim')
    fig.tight_layout()
    plt.show()

if True:
    from matplotlib import gridspec

    gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,3])
    ax = plt.subplot(gs[1,0])
    axr = plt.subplot(gs[1,1], sharey=ax)
    axt = plt.subplot(gs[0,0], sharex=ax)

    bins = np.linspace(-18 * scale_factor,18 * scale_factor, 50)
    ax.hist2d(complete_structure[:,ncbeads,axes[sel_axes][0]].ravel(),
              complete_structure[:,ncbeads,axes[sel_axes][1]].ravel(),
              bins=bins)
    i = np.random.permutation(np.prod(complete_structure[:,sel_beads].shape[:-1]))[:1000]
    ax.scatter(complete_structure[:,sel_beads,axes[sel_axes][0]].ravel()[i],
               complete_structure[:,sel_beads,axes[sel_axes][1]].ravel()[i],
               s=15,color='r',alpha=0.2)
    ax.set_xlim(-18 * scale_factor, 18 * scale_factor)
    ax.set_ylim(-18 * scale_factor, 18 * scale_factor)
    ax.set_aspect('equal')

    axt.hist(complete_structure[:,sel_beads,axes[sel_axes][0]].ravel(),
             bins=bins, histtype='step', color='r', normed=True)
    axt.hist(complete_structure[:,all_beads,axes[sel_axes][0]].ravel(),
             bins=bins, histtype='step', color='b', normed=True)
    axt.axvline(0.,ls='--',color='black',lw=2)
    plt.setp(axt.get_xticklabels(), visible=False)
    plt.setp(axt.get_yticklabels(), visible=False)
    axt.set_yticks(())
    axr.set_visible(False)

    ax.set_xlabel('{} coordinate [nm]'.format(sel_axes[0]))
    ax.set_ylabel('{} coordinate [nm]'.format(sel_axes[1]))

    plt.show()

if not True:
    gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,3])
    ax = plt.subplot(gs[1,0])
    axr = plt.subplot(gs[1,1])
    axt = plt.subplot(gs[0,0], sharex=ax)

    bins = np.linspace(0, 1, 20)
    # ax.hist2d(np.random.uniform(size=1000),
    #           np.random.uniform(size=1000), bins=bins)
    H, xedges, yedges = np.histogram2d(np.random.uniform(size=1000),
                                       np.random.uniform(size=1000),
                                       bins=bins)
    ax.imshow(H, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    plt.show()

if False:
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax1,ax2 = ax
    ax1.hist2d(rDNA_locus[:,:,axes[sel_axes][0]].ravel(),
               rDNA_locus[:,:,axes[sel_axes][1]].ravel(),
               bins=np.linspace(-18,18,n_bins))
    i = np.random.permutation(np.prod(complete_structure[:,sel_beads].shape[:-1]))[:1000]
    ax1.scatter(complete_structure[:,sel_beads,axes[0]].ravel()[i],
                complete_structure[:,sel_beads,axes[1]].ravel()[i],
                s=15,color='r',alpha=0.2)
    ax1.axvline(0.,ls='--',color='r',lw=2)
    if False:
        ax1.scatter(rDNA_locus[:,:,axes[0]].ravel(), rDNA_locus[:,:,axes[1]].ravel(),
                    s=1, color='r',alpha=0.1)
    ax2.hist(complete_structure[:,sel_beads,axes[0]].ravel(),bins=100)
    ax2.axvline(0.,ls='--',color='r',lw=2)
    plt.show()

print np.mean(complete_structure[:,sel_beads,axes[0]].ravel()>0)

mg_sel_beads = MultivariateGaussian()
mg_sel_beads.estimate(np.array(zip(complete_structure[:,sel_beads,axes[0]].ravel(),
                                   complete_structure[:,sel_beads,axes[1]].ravel())))
fit_str = "CM beads: mu ({:.2f},{:.2f}), sigma ({:.2f},{:.2f})"
print fit_str.format(mg_sel_beads.mu[0],  mg_sel_beads.mu[1],
                     mg_sel_beads.sigma[0,0], mg_sel_beads.sigma[1,1])
