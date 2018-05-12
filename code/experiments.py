import numpy as np
import rbm 
import util
import matplotlib.pyplot as plt

#Load data
global Xtrain, Xtest
Xtrain = np.load("../data/Xtrain.npy")
Xtest  = np.load("../data/Xtest.npy")


#Load parameters
global WP, WB, WC, d, k
WP = np.load("../models/MNISTWP.npy")
WB = np.load("../models/MNISTWB.npy")
WC = np.load("../models/MNISTWC.npy")

# WP = np.load("../data/WP_learnt.npy")
# WB = np.load("../data/WB_learnt.npy")
# WC = np.load("../data/WC_learnt.npy")

d = len(WC[0])
k = len(WB[0])
#Create RBM
mnistRBM  = rbm.rbm()
mnistRBM.set_params(WP,WB,WC)

#Number of Iterations
iterations = 500
num_chains = 100

#Gibbs sampler with single chain

x_samples, _, h_samples = mnistRBM.single_gibbs_sampler(X=Xtrain[0].reshape((1,-1)), iterations=iterations, WP=WP, WB=WB, WC=WC)

energies = np.empty(iterations)

for iteration in range(iterations):
    energies[iteration] = mnistRBM.energy(X=x_samples[iteration], H= h_samples[iteration], WP=WP, WB=WB, WC=WC) 

#Q3.1
plt.figure(1)
util.plot_im_array2(x_samples[-100:],28,100,10,"Test Sample")
plt.savefig('../figures/3_1.png')
#Q3.2
plt.figure(2)
plt.plot(energies)
plt.ylabel('Energy')
plt.xlabel('Sample No.')
plt.savefig('../figures/3_2.png')

# discarding the first 100 samples and calculating the autocorrelation

autocorrealtion = util.acorr(energies[100:])

#Q3.3
plt.figure(3)
plt.plot(autocorrealtion)
plt.ylabel('Autocorrealtion')
plt.xlabel('Sample No.')
plt.savefig('../figures/3_3.png')
#plt.show()

#Create multiple gibbs sampler chains
aggregate_x_samples = np.empty([iterations, num_chains, len(WC[0])])
aggreggate_h_samples =  np.empty([iterations, num_chains, len(WB[0])])
aggregate_energies = np.empty([iterations, num_chains])

aggregate_x_samples, _, aggreggate_h_samples = mnistRBM.single_gibbs_sampler(X=Xtrain[:num_chains], iterations=iterations, WP=WP, WB=WB, WC=WC)


#compute energies

for iteration in range(iterations):
    for chain in range(num_chains):
        aggregate_energies[iteration,chain] = mnistRBM.energy(X=aggregate_x_samples[iteration, chain].reshape(1,-1), H= aggreggate_h_samples[iteration, chain].reshape(1,-1), WP=WP, WB=WB, WC=WC)

#Q4.1 
plt.figure(4)
util.plot_im_array2(Xtrain,28,100,10,"Original Images")
plt.figure(5)
util.plot_im_array2(aggregate_x_samples[-1,:],28,100,10,"Sampled Images")
plt.savefig('../figures/4_1.png')

#Q4.2
plt.figure(6)
plt.plot(range(iterations), aggregate_energies[:,0], 'r', range(iterations), aggregate_energies[:,1], 'g', range(iterations), aggregate_energies[:,2],'b', range(iterations),aggregate_energies[:,3],'purple', range(iterations),aggregate_energies[:,4],'y')
#plt.plot(range(iterations), aggregate_energies[0,:], 'r', range(iterations), aggregate_energies[1,:], 'g', range(iterations), aggregate_energies[2,:],'b', range(iterations),aggregate_energies[3,:],'purple', range(iterations),aggregate_energies[4,:],'y')

plt.ylabel('Energy')
plt.savefig('../figures/4_2.png')
#plt.show()
print 'starting fitting operation'
mnistRBM.fit(X=Xtrain, K = 400)
print 'fitting done'

iterations = 500
num_chains = 100

# #Create multiple gibbs sampler chains
# aggregate_x_samples = np.empty([iterations, num_chains, len(WC[0])])
# aggregate_energies = np.empty([iterations, num_chains])
# aggregate_pxgh = np.empty([iterations, num_chains, len(WC[0])])

# aggregate_x_samples, aggregate_pxgh, aggreggate_h_samples = mnistRBM.single_gibbs_sampler(X=Xtrain[:num_chains], iterations=iterations, WP=WP, WB=WB, WC=WC)
x_samples = np.load('x_samples_of_last_iteration.npy')
x_probabilities = np.load('pxgh_of_last_iteration.npy')

#Q6.1 

plt.figure(7)
util.plot_im_array2(x_samples,28,100,10,"Sampled Images")
plt.savefig('../figures/6_1.png')


#Q6.2
plt.figure(8)
util.plot_im_array2(x_probabilities,28,100,10,"Probabilites as images")
plt.savefig('../figures/6_2.png')
