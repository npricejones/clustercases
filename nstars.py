"""stars2clusters - compute the number of clusters sampled by a number of stars assumng power law CMF and IMF                                                                                                              
Usage:                                                                                                         
    stars2clusters [-h] [-n NUMSTR] [-q SAMPLE] [-s STRMASS] [-c CLSMASS] [-a CLSIND] [-b STRIND] [-v AVOLUME] [-o OVOLUME]                     

Options:                                                                                                                 
    -h, --help                              Show this screen   
    -n NUMSTR, --numstr NUMSTR              Total number of stars observed [default: 1e5]                                                
    -s STRMASS, --strmass STRMASS           Limits of stellar mass function [default: (0.1,5.)]                
    -c CLSMASS, --clsmass CLSMASS           Limits of cluster mass function [default: (50,1e7)]
    -b STRIND, --strind STRIND              Index of stellar mass function [default: 2.65]                   
    -a CLSIND, --clsind CLSIND              Index of cluster mass function [default: 2.1]   
    -v AVOLUME, --annvol AVOLUME            Annulus volume in cubic kpc [default: 300]
    -o OVOLUME, --obsvol OVOLUME            Observed volume in cubic kpc [default: 30]            

Examples:
    python stars2clusters -s (1,10) 
"""

from docopt import docopt
import astropy.units as u
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
from tqdm import tqdm

arguments = docopt(__doc__)

# Read in arguments

# Number of stars observed
Nstr = int(float(arguments['--numstr']))
# Limits of stellar mass function
strlim = np.array(literal_eval(arguments['--strmass'])).astype(float)
# Limits of cluster mass function
clslim = np.array(literal_eval(arguments['--clsmass'])).astype(float)
# Power law index of stellar mass function (deprecated now that I am using broken power law)
strind = float(arguments['--strind'])
# Power law index of cluster mass function
clsind = float(arguments['--clsind'])
# Volume from which cluster members could be sampled
annvol = float(arguments['--annvol'])
# Volume from which stars are observed
obsvol = float(arguments['--obsvol'])

class mass_function_PL(object):
    """
    Convenience class for power law operations. Perhaps should be deprecated to avoid trouble with integer power laws
    """
    def __init__(self,ind,minmass,maxmass):
        self.ind = ind
        self.min = minmass
        self.max = maxmass

    def normcount(self,total):
        self.norm = (total*(self.ind-1))*(self.min**(-self.ind+1)-self.max**(-self.ind+1))**-1

    def normmass(self,total):
        self.norm = (total*(self.ind-2))*(self.min**(-self.ind+2)-self.max**(-self.ind+2))**-1

    def count(self,lower,upper):
        integrate = (lower**(-self.ind+1)-upper**(-self.ind+1))
        return np.round((self.norm/(self.ind-1))*integrate)

    def mass(self,lower,upper):
        integrate = (lower**(-self.ind+2)-upper**(-self.ind+2))
        return (self.norm/(self.ind-2))*integrate

# Determine conversion factor that turns a number of stars into a mass
Ml, Mu = strlim
Mm = 0.5
alpha1 = 1.3
alpha2 = 2.3
# M = CN
Cnum = ((1./(alpha1-2.))*((Ml)**(2-alpha1) - (Mm)**(2-alpha1))) + ((1./(alpha2-2.))*((Mm)**(2-alpha2) - (Mu)**(2-alpha2)))
Cden = ((1./(alpha1-1.))*((Ml)**(1-alpha1) - (Mm)**(1-alpha1))) + ((1./(alpha2-1.))*((Mm)**(1-alpha2) - (Mu)**(1-alpha2)))
C = (Cnum/Cden)

# Determine mass of stars observed
Mstr = C*Nstr

# Calculate the average stellar density of the Milky Way (extremely rough)
den = 6e10/(np.pi*20**2)

# Determine the sampled mass and the observed mass
annmass = den*annvol
obsmass = den*obsvol

#dummyN = int(1e3*Nstr)

# Calculate the sampling rate
S = Mstr/annmass

# Determine the number of clusters that could potentially be sampled
cmf = mass_function_PL(clsind,50.,1e7)
cmf.normmass(annmass)
Ncls = cmf.count(clslim[0],clslim[1])

steps = int(1e7)
# Calculate the minimum possible observed mass with this sampling distribution
minMass = (C/S)
print('minmass {0}'.format(minMass))
# Measure the probability of observing a cluster with a given mass
M = np.linspace(np.log10(minMass),np.log10(clslim[1]),steps)
P = (10**M)**(1-clsind)
# Normalize probability
Psum = np.sum(P)
P = P/Psum

trueclusters = np.array([],dtype=int)
clusters = np.array([],dtype=int)
it = 0
while np.sum(clusters) < Nstr:
    print(it,np.sum(clusters))
    size = 10**np.random.choice(M,size=int(1e6),p=P)
    members = np.random.poisson(size/C)
    print(members*S)
    trueclusters = np.append(trueclusters,members)
    clusters = np.append(clusters,np.floor(members*S))
    it+=1


# # Pick the sizes of cluster to observe
# logsizes = np.random.choice(M,size=dummyN,p=P)
# sizes = 10**logsizes/C

# # Calculate the number of stars observed per cluster
# newstars = np.random.poisson(sizes).astype(int)
# newstars = np.floor(S*newstars)
# totalstars = np.cumsum(newstars)

# # Find where the total number of stars observed exceeds the required amount
# exceedind = np.where(totalstars >= Nstr)[0][0]-1
# clusters = newstars[:exceedind]

# # Add one cheat cluster to fill the gap
# clusters = np.append(clusters,Nstr-np.sum(clusters))
# # Remove clusters with no sampled stars
# newclusters = clusters[clusters>0]
np.savetxt('stararray.txt',clusters,header="cmin{0}_cmax{1}_cind{2}_totalclusters{3}".format(clslim[0],clslim[1],clsind,len(clusters)))

