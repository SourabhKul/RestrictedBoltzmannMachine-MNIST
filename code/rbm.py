import numpy as np
import util
import matplotlib.pyplot as plt

class rbm():

    def __init__(self):
        '''
        function __init__()
        inputs: None
        outputs: None
        '''
        #np.random.seed(45) 
        pass

    def sigmoid(self,x): 
        return 1/(1+np.exp(-x))

    def set_params(self, WP, WB, WC):
        '''
        function set_params(self, WP, WB, WC)

        Description: This function sets member variables for storing
        the RBM model parameters.

        inputs: WP - RBM pairwise parameters. DxK array.
                WB - RBM hidden biases. 1xK array.
                WC - RBM visible biases. 1xD array.
        outputs: None
        '''
        self.WP = WP
        self.WB = WB
        self.WC = WC
        
        pass

    def get_params(self):
        '''
        function get_params(self)

        Description: This function gets the member variables that store
        the RBM model parameters.

        inputs: none.

        outputs: (WP, WB, WC)
                WP - RBM pairwise parameters. DxK array.
                WB - RBM hidden biases. 1xK array.
                WC - RBM visible biases. 1xD array.
        '''

        return (self.WP, self.WB, self.WC)

    def energy(self,X,H,WP=None,WB=None,WC=None ):
        '''
        function energy(self,X,H,WP=None,WB=None,WC=None )

        Description: This function computes the joint energy
        of the RBM given parameters WP, WB, WC, a visible
        vector X, and a hidden vecotor H. When the parameters
        are not specified as inputs, the current values
        of the member variables storing the parameters
        should be used.

        inputs: X  - Visible vector. 1xD array.
                H  - Hidden vector. 1xK array.
                WP - RBM pairwise parameters. DxK array.
                WB - RBM hidden biases. 1xK array.
                WC - RBM visible biases. 1xD array.

        outputs: Value of the energy function. Scalar float.
        '''
        if WP is None: WP = self.WP
        if WB is None: WB = self.WB
        if WC is None: WC = self.WC
        
        #energy = -np.sum( WP*np.outer(X,H) + np.dot(WB, H) + np.dot(WC,X))
        energy_1 = -np.sum(WP*np.dot(X.T,H))
        energy_2 = -np.sum(WB*H) 
        energy_3 = -np.sum(WC*X)

        return float(energy_1+energy_2+energy_3)

    def phgx(self,X,WP=None,WB=None):
        '''
        function phgx(self,X,WP=None,WB=None)

        Description: This function computes P(H_k=1|X)
        for all k=1:K given the visible vector X and
        the RBM model parameters WP and WB. When the
        parameters are not specified as inputs, the
        current values of the member variables storing
        the parameters should be used.

        inputs: X  - Visible vector. 1xD array.
                WP - RBM pairwise parameters. DxK array.
                WB - RBM hidden biases. 1xK array.

        outputs: Array of shape 1xK where element k
                gives P(H_k=1|X).
        '''

        if WP is None: WP = self.WP
        if WB is None: WB = self.WB
        
        phgx = self.sigmoid(WB + np.dot(X, WP))
        
        return phgx

    def pxgh(self,H,WP=None,WC=None):
        '''
        function pxgh(self,H,WP=None,WC=None)

        Description: This function computes P(X_d=1|H)
        for all d=1:D given the hidden vector H and
        the RBM model parameters WP and WC. When the
        parameters are not specified as inputs, the
        current values of the member variables storing
        the parameters should be used.

        inputs: H  - Hidden vector. 1xK array.
                WP - RBM pairwise parameters. DxK array.
                WC - RBM visible biases. 1xD array.

        outputs: Array of shape 1xD where element d
                gives P(X_d=1|H).
        '''

        if WP is None: WP = self.WP
        if WC is None: WC = self.WC
        
        pxgh = self.sigmoid(WC + np.dot(H, WP.T))

        return pxgh

    def gibbs_x_step(self,H,WP=None,WC=None):
        '''
        function gibbs_x_step(self,H,WP=None,WC=None)

        Description: This function draws a sample for
        the visible vector X conditional on the specified
        hidden vector H and the model parameters WP and WC.
        When the parameters are not specified as inputs, the
        current values of the member variables storing
        the parameters should be used.

        inputs: H  - Hidden vector. 1xK array.
                WP - RBM pairwise parameters. DxK array.
                WC - RBM visible biases. 1xD array.

        outputs: A tuple of arrays (X, PX) where
        X is an array of shape 1xD where element d
        is a sample from P(X_d|H), and PX is an array of
        shape 1xD where element d is P(X_d|H).
        '''

        if WP is None: WP = self.WP
        if WC is None: WC = self.WC

        pxgh = self.pxgh(H=H, WP=WP, WC=WC)
        x_sample = np.where(np.random.uniform(0, 1, pxgh.shape) < pxgh, 1.0, 0.0)
        
        return (x_sample, pxgh)

    def gibbs_h_step(self,X,WP=None,WB=None):
        '''
        function gibbs_h_step(self,X,WP=None,WB=None)

        Description: This function draws a sample for
        the hidden vector H conditional on the specified
        visible vector X and the model parameters WP and WC.
        When the parameters are not specified as inputs, the
        current values of the member variables storing
        the parameters should be used.

        inputs: X  - Visible vector. 1xD array.
                WP - RBM pairwise parameters. DxK array.
                WB - RBM hidden biases. 1xK array.

        outputs: A tuple of arrays (H, PH) where
        H is an array of shape 1xK where element k
        is a sample from P(H_k|X), and PH is an array of
        shape 1xK where element k is P(H_k|X).
        '''

        if WP is None: WP = self.WP
        if WB is None: WB = self.WB

        phgx = self.phgx(X=X, WP=WP, WB=WB)
        h_sample = np.where(np.random.uniform(0,1, phgx.shape) < phgx, 1.0, 0.0)
        
        return (h_sample, phgx)

    def grad(self,X,XS,lam,WP=None,WB=None,WC=None):
        '''
        function grad(self,X,XS,lam,WP=None,WB=None,WC=None)

        Description: This function computes the gradients
        for all RBM parameters given real visible vectors
        X, samples of visible vectors XS, and the current
        RBM model parameters. When the parameters are not
        specified as inputs, the current values of the
        member variables storing the parameters should be
        used.

        inputs: X  - Real visible vectors. NxD array.
                XS - Sampled visible vectors. SxD array.
                lam - L2 regularization strength. Scalar float.
                WP - RBM pairwise parameters. DxK array.
                WB - RBM hidden biases. 1xK array.
                WC - RBM visible biases. 1xD array.

        outputs: A three element tuple (gWP, gWB, gWC)
        where gWP is the gradient wrt WP (DxK array),
        gWB is the gradient wrt WB (1xK array), and
        gWC is the gradient wrt WC (1xD array).
        '''


        if WP is None: WP = self.WP
        if WB is None: WB = self.WB
        if WC is None: WC = self.WC
        

        
        N,D = X.shape
        S,_ = XS.shape
        p_h_o = self.phgx(X=X, WP=WP, WB=WB)
        p_h_s = self.phgx(X=XS,  WP=WP, WB=WB)
                
        gWP = np.dot(X.T,p_h_o)/float(N) - np.dot(XS.T,p_h_s)/float(S) - lam*WP
        gWB = np.sum(p_h_o,axis=0)/float(N) - np.sum(p_h_s,axis=0)/float(S) - lam*WB
        gWC = np.sum(X,axis=0)/float(N) - np.sum(XS,axis=0)/float(S) - lam*WC

        # gWP = np.dot(X.T,p_h_o)/N - np.dot(X.T,p_h_s)/S - lam*WP
        # gWB = np.sum(p_h_o)/N - np.sum(p_h_s)/S - lam*WB
        # gWC = np.sum(X)/N - np.sum(XS)/S - lam*WC
        
        return (gWP, gWB, gWC)
    
    def single_gibbs_sampler(self, X, iterations=500, WP=None, WB=None, WC=None):
        if WP is None: WP = self.WP
        if WB is None: WB = self.WB
        if WC is None: WC = self.WC
        
        N, _ = X.shape
        
        x_samples = np.empty([iterations, N, len(WC[0])])
        pxgh = np.empty([iterations, N, len(WC[0])])
        h_samples = np.empty([iterations, N, len(WB[0])])
        phgx = np.empty([iterations, N, len(WB[0])])
        

        #initialize hidden layer to random values? No.
        #h_samples[0] = np.random.binomial(1, 0.5, len(WB))
        
        x_samples[0] = X
        (h_samples[0], phgx[0]) = self.gibbs_h_step(X=x_samples[0], WP=WP, WB=WB)
        #h_reshaped = h_samples[0].reshape((1,-1))
        #x_reshaped = x_samples[0].reshape((1,-1))
        h_reshaped = h_samples[0]
        x_reshaped = x_samples[0]
                
        #Begin sampling
        for iteration in range(1,iterations):
                #h_reshaped = h_samples[iteration-1].reshape((1,-1))
                h_reshaped = h_samples[iteration-1]
                #x-step
                (x_samples[iteration], pxgh[iteration]) = self.gibbs_x_step(H=h_reshaped, WP=WP, WC=WC)
                #h-step
                #x_reshaped = x_samples[iteration].reshape((1,-1))
                x_reshaped = x_samples[iteration]
                (h_samples[iteration], phgx[iteration]) = self.gibbs_h_step(X=x_reshaped, WP=WP, WB=WB)
                
        return x_samples, pxgh, h_samples

    def fit(self, X, K = 100, C = 100, B=100, eps=0.01, lam=0.01, iters=10000,WP0=None,WB0=None,WC0=None):

        '''
        function learn(self, X, K = 100, C = 100, B=100, eps=0.01, lam=0.01, iters=10000,WP0=None,WB0=None,WC0=None)

        Description: This function learns the RBM model
        using mini-batch stochastic maximum likelihood
        given a data set X of visible vectors. If the
        initial RBM parameters WP0, WB0, WC0 are None,
        initialize by sampling from a zero-mean, unit std
        normal distribution.

        inputs: X  - Real visible vectors. NxD array.
                K  - Number of hidden units.  Scalar int.
                C  - Number of parallel chains  to use. Scalar int.
                B  - Batch size to use. Scalar int.
                eps - Learning rate. Scalar float.
                lam - L2 regularization parameter. Scalar float.
                iters - Number of optimization iterations. Scalar int.
                WP0 - Initial RBM pairwise parameters. DxK array.
                WB0 - Initial RBM hidden biases. 1xK array.
                WC0 - Initial RBM visible biases. 1xD array.

        outputs: None. The final learned parameter values
        should be stored in the WP, WB, and WC member variables.
        '''
        N, D = X.shape
        #minibatch = X[np.random.randint(0,X.shape[0],B)]
        
        WP = WP0
        WB = WB0
        WC = WC0

        if WP0 is None: WP = np.random.normal(0,1,[D,K])
        if WB0 is None: WB = np.random.normal(0,1,[1,K])
        if WC0 is None: WC = np.random.normal(0,1,[1,D])
        
        
        aggregate_x_samples = np.empty([iters, C, D])
        aggregate_pxgh = np.empty([iters, C,D])
        aggregate_x_samples[0] = np.random.randint(0,1,[C,D])
        

        
        for iter in range(1,iters):
                print iter
                minibatch = X[np.random.randint(0,X.shape[0],B)]
                #sampling
                
                h, _ = self.gibbs_h_step(X=aggregate_x_samples[iter-1], WP=WP, WB=WB)
                aggregate_x_samples[iter], aggregate_pxgh[iter] = self.gibbs_x_step(H = h,WP=WP, WC=WC)
                #update
                
                (gWP, gWB, gWC) = self.grad(X=minibatch, XS =aggregate_x_samples[iter], lam=lam, WP=WP, WB=WB, WC=WC)  
                WP += eps*gWP
                WB += eps*gWB
                WC += eps*gWC 

                self.set_params(WP=WP, WB=WB, WC=WC)
        
        np.save('x_samples_of_last_iteration', aggregate_x_samples[iters-1])
        np.save('pxgh_of_last_iteration', aggregate_pxgh[iters-1])
        # np.save('../data/WP_learnt', WP)
        # np.save('../data/WB_learnt', WB)
        # np.save('../data/WC_learnt', WC)
        pass
