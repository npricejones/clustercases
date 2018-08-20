import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import euclidean_distances
default_cmap='viridis' # for 2D histograms                                     
fs=12

def readclusterlabels(fname,timestamp,
                      genfn='normalgeneration'):
    """
    Read file with cluster information.
    """
    

def membercount(labels):
    """
    For a list of labels, return the number of objects with each label.

    labels:   label for each object of interest

    Returns membership for each label and the corresponding labels.
    """
    ulab = np.unique(labels)
    members = np.zeros(len(ulab))
    for u in range(len(ulab)):
        members[u] = len(np.where(labels==ulab[u])[0])
    return members,ulab

def sortmembercount(labels):
    """
    For a list of labels, return the number of objects with each label sorted
    so the largest group is first.        
                                                                               
    labels:   label for each object of interest                                
                                                                            
    Returns sorted membership for each label and the corresponding labels. 
    """
    ulab = np.unique(labels)
    members = np.zeros(len(ulab))
    for u in range(len(ulab)):
        members[u] = len(np.where(labels==ulab[u])[0])
    sizelist = np.argsort(members)[::-1]
    return members[sizelist],ulab[sizelist]

def crossmatch(labels_pred,labels_true,minmembers=1):
    """
    Match each found cluster to the original cluster that contributed the 
    majority of its members.

    labels_pred:   Labels for each object matching to found clusters.
    labels_true:   Labels for each object matching to the original clusters.
    minmembers:   Only consider clusters larger than the given size.

    Returns list of found labels and matched true labels.
    """
    plabs = (np.unique(labels_pred)).astype(int)
    bad = np.where(plabs==-1)
    if len(bad[0])>0:
        plabs = np.delete(plabs,bad[0][0])
    tlabs = -np.ones(plabs.shape,dtype=int)
    for p in range(len(plabs)):
        predmatch = np.where(labels_pred==plabs[p])
        if len(labels_pred[predmatch])>=minmembers:
            truepredmatch = labels_true[predmatch]
            truecounts,trueinds = sortmembercount(truepredmatch)
            tlabs[p]=trueinds[0] #stands for majority stakeholder
    return plabs,tlabs
    
def efficiency_completeness(labels_pred,
                            labels_true,
                            minmembers=1):
    """
    Compute the efficiency and completeness for each cluster in a sample.

    labels_pred:   Labels for each object matching to found clusters.         
    labels_true:   Labels for each object matching to the original clusters.    
    minmembers:   Only consider clusters larger than the given size.  
    
    Returns lists of efficiency, completeness, found labels and 
    matched true labels.
    """
    plabs,tlabs = crossmatch(labels_pred,labels_true,minmembers=minmembers)
    bad = np.where(plabs==-1)
    if len(bad[0])>0:
        plabs = np.delete(plabs,bad[0][0])
    # Initialize output as disallowed values.
    efficiency = -np.ones(plabs.shape)
    completeness = -np.ones(plabs.shape)
    # Go through all found clusters
    for p in range(len(plabs)):
        predmatch = np.where(labels_pred==plabs[p])
        pred_pred = labels_pred[predmatch]
        true_pred = labels_true[predmatch]
        # Proceed if cluster is large enough
        if len(pred_pred)>=minmembers:
            # Find all members of the matched original cluster
            truematch = np.where(labels_true==tlabs[p])
            true_true = labels_true[truematch]
            # Find all members of the matched original cluster in the 
            # found cluster
            predtruematch = np.where((labels_true==tlabs[p]) & (labels_pred==plabs[p]))
            pred_predtrue = labels_pred[predtruematch]
            # Find the number of stars in the found cluster
            Nrecover = len(pred_pred)
            # Find the number of stars in the original cluster
            Noriginal = len(true_true)
            # Find the number of stars of the original cluster in the 
            # found cluster
            Nmajority = len(pred_predtrue)
            # Find efficiency and completeness
            efficiency[p] = Nmajority/Nrecover
            completeness[p] = Nmajority/Noriginal
    return efficiency, completeness, plabs, tlabs

def size_distribution(labels_pred,
                      labels_true,
                      runlabel,
                      nbins=20,
                      sizebins = [],
                      logval=True,
                      normval=False,
                      figsize=(15,8)):
    tcount,tlabs = membercount(labels_true)
    pcount,plabs = membercount(labels_pred)
    bad = np.where(plabs==-1)
    if len(bad[0])>0:
        plabs = np.delete(plabs,bad[0][0])
        pcount = np.delete(pcount,bad[0][0])
    if sizebins == []:
        sizebins = np.linspace(1,max(max(pcount),
                                     max(tcount)),nbins)
    matchplabs,matchtlabs = crossmatch(labels_pred,
                                       labels_true,
                                       minmembers=1)
    logval=True
    normval=False
    plt.figure(figsize=figsize)
    plt.suptitle(runlabel)
    plt.subplot(121)
    plt.hist(tcount,
             bins=sizebins,
             alpha=0.2,
             linewidth=2,
             color='k',
             edgecolor='k',
             log=logval,
             label='True distribution',
             normed=normval)
    plt.hist(pcount,
             bins=sizebins,
             alpha=0.5,
             linewidth=2,
             edgecolor='C0',
             log=logval,
             label='Predicted distribution',
             normed=normval)
    plt.legend(loc='best')
    plt.xlabel('clustermembers')
    plt.ylabel('count')
    
    plt.subplot(122)
    plt.hist(tcount,
             bins=sizebins,
             alpha=0.2,
             linewidth=2,
             color='k',
             edgecolor='k',
             log=logval,
             label='True distribution',
             normed=normval)
    plt.hist(tcount[matchtlabs],
             bins=sizebins,
             alpha=0.5,
             linewidth=2,
             color='C1',
             edgecolor='C1',
             log=logval,
             label='Matched distribution',
             normed=normval)
    plt.legend(loc='best')
    plt.xlabel('cluster members')
    plt.ylabel('count')

def sizebinned_EChists(labels_pred,
                       labels_true,
                       runlabel,
                       memberbins=[],
                       nbins=15,alpha=0.7,
                       lw=3,figsize=(15,10)):
    """
    For a given set of true and predicted labels, plot the distributions of efficiency and completeness, divided by cluster size according to memberbins.
    
    labels_pred:   Labels for each object matching to found clusters.
    labels_true:   Labels for each object matching to the original clusters.
    runlabel:      Supertitle for the plot
    memberbins:    List of tuples to divide up the clusters by size, defaults to [(1,5),(5,10),(10,50),(50,100),(100,max(max(found cluster sizes),max(known cluster sizes))]
    nbins:         Number of bins to use for efficiency and completeness values.
    alpha:         Opacity of histogram lines
    lw:            Width of histogram lines
    figsize:       Size of the figure.
    
    """
    tcount,tlabs = membercount(labels_true)
    pcount,plabs = membercount(labels_pred)
    # Discount outlier classification
    bad = np.where(plabs==-1)
    if len(bad[0])>0:
        plabs = np.delete(plabs,bad[0][0])
        pcount = np.delete(pcount,bad[0][0])
    efficiency,completeness,matchplabs,matchtlabs = efficiency_completeness(labels_pred,labels_true,minmembers=1)
    
    if memberbins == []:
        memberbins = [(1,5),(5,10),(10,50),(50,100),(100,int(max(max(pcount),max(tcount))))]
    bins = np.linspace(0,1,nbins)
    plt.figure(figsize=figsize)
    plt.suptitle(runlabel)
    for size in range(len(memberbins)):
        minm = memberbins[size][0]
        maxm = memberbins[size][1]
        psizematch = np.where((pcount>=minm) & (pcount<maxm))
        tsizematch = np.where((tcount[matchtlabs]>=minm) & (tcount[matchtlabs]<maxm))      
        plt.subplot(221)
        if psizematch[0] != []:
            plt.hist(efficiency[psizematch],
                     bins=bins,
                     normed=True,
                     log=False,
                     alpha=alpha,
                     linewidth=lw,
                     facecolor='none',
                     edgecolor='C{0}'.format(size),
                     label='{0} clusters with\n{1} to {2} members'.format(len(psizematch[0]),minm,maxm))
        plt.xlabel('efficiency')
        plt.ylabel('probability')
        plt.subplot(222)
        if psizematch[0] != []:
            plt.hist(completeness[psizematch],
                     bins=bins,
                     normed=True,
                     log=False,
                     alpha=alpha,
                     linewidth=lw,
                     facecolor='none',
                     edgecolor='C{0}'.format(size))
        plt.xlabel('completeness')
        plt.ylabel('probability')
        plt.subplot(223)
        if tsizematch[0] != []:
            plt.hist(efficiency[tsizematch],
                     bins=bins,
                     normed=True,
                     log=False,
                     alpha=alpha,
                     linewidth=lw,
                     facecolor='none',
                     edgecolor='C{0}'.format(size),
                     label='{0} clusters with\n{1} to {2} members'.format(len(tsizematch[0]),minm,maxm))
        plt.xlabel('efficiency')
        plt.ylabel('probability')
        plt.subplot(224)
        if tsizematch[0] != []:
            plt.hist(completeness[tsizematch],
                     bins=bins,
                     normed=True,
                     log=False,
                     alpha=alpha,
                     linewidth=lw,
                     facecolor='none',
                     edgecolor='C{0}'.format(size))
        plt.xlabel('completeness')
        plt.ylabel('probability')
    plt.subplot(221)
    legend = plt.legend(loc='best',title='found clusters')
    legend.get_frame().set_linewidth(0.0)
    plt.subplot(223)
    legend = plt.legend(loc='best',title='known matched clusters')
    legend.get_frame().set_linewidth(0.0)

class distance_metrics(object):
    def __init__(self,spectra,**kwargs):
        self.spectra = spectra
        self.stree = KDTree(spectra,**kwargs)
        
    def core_border_frac(self,labels_pred,cbn):
        pcount,plabs = membercount(labels_pred)
        # Discount outlier classification
        bad = np.where(plabs==-1)
        if len(bad[0])>0:
            plabs = np.delete(plabs,bad[0][0])
            pcount = np.delete(pcount,bad[0][0])
        core_border = -np.ones(len(plabs))
        for p in range(len(plabs)):
            meminds = np.where(labels_pred==plabs[p])
            total = pcount[p]
            cores = len(np.where(cbn[meminds]==1)[0])
            core_border[p] = (2*cores-total)/total
        return core_border   
    
    def _central_tree(self,labels_pred,**kwargs):
        """
        Make tree out of 'central' spectra
        """
        pcount,plabs = membercount(labels_pred)
        # Discount outlier classification
        bad = np.where(plabs==-1)
        if len(bad[0])>0:
            plabs = np.delete(plabs,bad[0][0])
            pcount = np.delete(pcount,bad[0][0])
        central = np.zeros((len(plabs),self.spectra.shape[1]))
        for p in range(len(plabs)):
            meminds = np.where(labels_pred==plabs[p])
            members = self.spectra[meminds]
            central[p] = np.mean(members)
        return central,KDTree(central,**kwargs)
        
    def silhouette(self,labels_pred,k=10,pwmetric=euclidean_distances,**kwargs):
        """
        labels_pred:   Labels for each object matching to found clusters.
        k:             Number of nearest clusters to consider (minimum 2)
        pwmetric:      Function to find pairwise distances
        **kwargs:      kwargs for sklearn.neighbors.KDTree
        """
        central,ctree = self._central_tree(labels_pred,**kwargs)
        dist,inds = ctree.query(central,k=k)
        pcount,plabs = membercount(labels_pred)
        # Discount outlier classification
        bad = np.where(plabs==-1)
        if len(bad[0])>0:
            plabs = np.delete(plabs,bad[0][0])
            pcount = np.delete(pcount,bad[0][0])
        standards = np.zeros(len(plabs))
        extremes = np.zeros(len(plabs))
        for p in range(len(plabs)):
            meminds = np.where(labels_pred==plabs[p])
            members = self.spectra[meminds]
            nearby = inds[p]
            nonmembers=np.empty(0)
            for n in range(1,len(nearby)):
                plab = nearby[n]
                meminds = np.where(labels_pred==plab)
                if n==1:
                    nonmembers = self.spectra[meminds]
                elif n!=1:
                    nonmembers = np.append(nonmembers,self.spectra[meminds],axis=0)
            memdist = euclidean_distances(members,members)
            nonmemdist = euclidean_distances(members,nonmembers)

            stintra = np.median(memdist)
            exintra = np.max(memdist)
            stinter = np.median(nonmemdist)
            exinter = np.min(nonmemdist)
            standards[p] = (stinter-stintra)/np.max([stinter,stintra])
            extremes[p] = (exinter-exintra)/np.max([exinter,exintra])
        return standards,extremes
