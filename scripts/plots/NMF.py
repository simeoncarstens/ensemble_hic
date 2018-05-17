if False:
    mds = array(map(fwm_eval, ens_flatter))
    d = FWM.data_points
    a = mds.reshape(-1, n_structures, len(d))

    from sklearn.decomposition import NMF
    model = NMF(n_components=2,
                init='random', random_state=0, tol=1e-10,
                max_iter=1000)
    model.verbose=0
    W = model.fit_transform(a.reshape(-1,len(d)))
    H = model.components_
    
    prob = (W.T/W.sum(1)).T
    
    sep=np.fabs(np.dot(FWM.data_points[:,:2],np.array([1,-1])))
    
    threshold = 0.5
    c = (a.reshape(-1,len(d))>threshold).astype('i')

    a = g
    meens = 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    stds = a.reshape(-1,len(d)).std(0)
    means = a.reshape(-1,len(d)).mean(0)
    ax.plot((stds, means)[meens])
    markers = ('x','o','s')
    for i in range(0,3):
        sel = np.where(np.fabs(d[:,1]-d[:,0]) == i+2)[0]
        ax.scatter(sel, (stds, means)[meens][sel], marker=markers[i], color='black',
                    label='|i-j|={}'.format(i+2))
    ax.set_xlabel('data point')
    ax.set_ylabel(('std', 'mean')[meens] + '(mock data)')
    ax.legend()
    plt.show()
                    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = ax.scatter(sep, stds, c=log(d[:,2]+1))
    cb = plt.colorbar(m)
    cb.set_label('log(experimental count)')
    ax.set_xlabel('sequence separation')
    ax.set_ylabel('std(mock data)')
    plt.show()
