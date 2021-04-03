"""
MSTPPG: Marked Multivariate Point Process Generator
"""

import sys
import arrow
import utils
import numpy as np
from scipy.stats import norm

from numpy import power

class MaternKernel(object):
    """
    The Matérn covariance between two points separated by d distance units is given by 
    """
    def __init__(self, p=2, rho=1e-2, sigma_t=5e-2):
        self.p       = p
        self.rho     = rho
        self.sigma_t = sigma_t
    

    def nu(self, t, s, his_t, his_s):
        delta_t = t - his_t
        # term 1
        t1 = np.exp(-(np.sqrt(2*self.p+1)*delta_t)/self.rho)
        # term 2
        t2 = np.math.factorial(self.p)/np.math.factorial(2 * self.p)
        # term 3
        t3 = np.sum(np.array([np.math.factorial(self.p + i)
            /(np.math.factorial(i)*np.math.factorial(self.p - i)) 
            * power((2*np.sqrt(2*self.p + 1) * delta_t)/self.rho , self.p - i) 
            for i in range(self.p + 1)]))
        # print(self.sigma_t * t1 * t2 * t3)
        return self.sigma_t * t1 * t2 * t3


class ExpKernel(object):
    """
    Exponential Kernel function 
    """
    def __init__(self, Alpha=np.ones([2,2]), beta=1.):
        self.Alpha   = Alpha
        self.beta    = beta
        # self.sigma_x = sigma_x
        # self.sigma_y = sigma_y

    def nu(self, t, s, his_t, his_s):
        
        delta_t         = t - his_t
        alpha_s_his_s   = np.asarray([self.Alpha[int(s), int(h)] for h in his_s])
        # delta_s = s - his_s
        # delta_m         = np.abs(m - his_m)
        # delta_x = delta_s[:, 0]
        # delta_y = delta_s[:, 1]
        
        # print(alpha_s_his_s * \
        #     np.exp(- self.beta * delta_t))

        return alpha_s_his_s * \
            np.exp(- self.beta * delta_t)
            # (self.C / (2 * np.pi * self.sigma_x * self.sigma_y * delta_t)) * \
            # np.exp((- 1. / (2 * delta_t)) * \
                # ((np.square(delta_x) / np.square(self.sigma_x)) + \
                # (np.square(delta_y) / np.square(self.sigma_y))))


class HawkesLam(object):
    """Intensity of Marked Spatio-temporal Hawkes point process"""
    def __init__(self, mu, kernel, maximum=1e+4, threshold=5, inc=True):
        self.mu        = mu
        self.kernel    = kernel
        self.maximum   = maximum
        self.threshold = threshold
        self.inc       = inc

    def value(self, t, his_t, s, his_s):
        """
        return the intensity value at (t, s)
        The last element of seq_t, seq_s, seq_m is the location (t, s) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        """
        if len(his_t) > 0:
            if not self.inc:
                val = self.mu - np.sum(self.kernel.nu(t, s, his_t, his_s))
                if val < self.threshold:
                    val = self.threshold
            else:
                val = self.mu + np.sum(self.kernel.nu(t, s, his_t, his_s))
        else:
            val = self.mu
        return val

    def upper_bound(self):
        """return the upper bound of the intensity value"""
        return self.maximum

    def __str__(self):
        return "Hawkes processes"

class SelfCorrectKernel(object):
    """Kernel function for self-correcting process"""
    def __init__(self, alpha):
        self.alpha = alpha

    def nu(self, t, his_t):
        # delta_t = t - his_t
        return self.alpha * (np.ones(his_t.shape))
        # return self.alpha * delta_t

class SelfCorrectLam(object):
    """Intensity of self-correcting process"""
    def __init__(self, mu, kernel, maximum=1e+4):
        self.mu      = mu
        self.kernel  = kernel
        self.maximum = maximum

    def value(self, t, his_t, s, his_s):
        """
        return the intensity value at (t, s)
        The last element of seq_t, seq_s is the location (t, s) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        """

        if len(his_t) > 0:
            # print(np.prod(self.kernel.nu(t, his_t)))
            val = np.exp(self.mu * t - np.sum(self.kernel.nu(t, his_t)))
        else:
            val = np.exp(self.mu * t)
        return val
    
    def upper_bound(self):
        """return the upper bound of the intensity value"""
        return self.maximum

    def __str__(self):
        return "Self-correcting processes"


class SpatialTemporalPointProcess(object):
    """
    Marked Spatial Temporal Hawkes Process

    A stochastic marked spatial temporal points generator based on Hawkes process.
    """

    def __init__(self, lam):
        """
        Params:
        """
        # model parameters
        self.lam     = lam

    def _homogeneous_poisson_sampling(self, T=[0, 1], S=3):
        """
        To generate a homogeneous Poisson point pattern in space S X T, it basically
        takes two steps:
        1. Simulate the number of events n = N(S) occurring in S according to a
        Poisson distribution with mean lam * |S X T|.
        2. Sample each of the n location according to a uniform distribution on S
        respectively.

        Args:
            lam: intensity (or maximum intensity when used by thining algorithm)
            S:   [(min_t, max_t), (min_x, max_x), (min_m, max_m), ...] indicates the
                range of coordinates regarding a square (or cubic ...) region.
        Returns:
            samples: point process samples:
            [(t1, s1), (t2, s2), ..., (tn, sn)]
        """
        _S     = [T]
        
        # sample the number of events from S
        # what is lebesgue measure here for? Why the number of sample is determined by this
        n      = utils.lebesgue_measure(_S)
        N      = np.random.poisson(size=1, lam=self.lam.upper_bound() * n)

        # simulate spatial sequence and temporal sequence separately.
        points = [ np.random.uniform(_S[0][0], _S[0][1], N), 
                   np.random.randint(low=0, high=S, size=N)]

        points = np.array(points).transpose()
        # print(points)
        # sort the sequence regarding the ascending order of the temporal sample.
        points = points[points[:, 0].argsort()]
        return points

    def _inhomogeneous_poisson_thinning(self, homo_points, verbose):
        """
        To generate a realization of an inhomogeneous Poisson process in S × T, this
        function uses a thining algorithm as follows. For a given intensity function
        lam(s, t):
        1. Define an upper bound max_lam for the intensity function lam(s, t)
        2. Simulate a homogeneous Poisson process with intensity max_lam.
        3. "Thin" the simulated process as follows,
            a. Compute p = lam(s, t)/max_lam for each point (s, t) of the homogeneous
            Poisson process
            b. Generate a sample u from the uniform distribution on (0, 1)
            c. Retain the locations for which u <= p.
        """
        retained_points = np.empty((0, homo_points.shape[1]))
        if verbose:
            print("[%s] generate %s samples from homogeneous poisson point process" % \
                (arrow.now(), homo_points.shape), file=sys.stderr)
        # thining samples by acceptance rate.
        for i in range(homo_points.shape[0]):
            # current time, location and generated historical times and locations.
            t     = homo_points[i, 0]
            s     = homo_points[i, 1]
            his_t = retained_points[:, 0]
            his_s = retained_points[:, 1]
            
            # thinning
            lam_value = self.lam.value(t, his_t, s, his_s)
            lam_bar   = self.lam.upper_bound()
            D         = np.random.uniform()
            # - if lam_value is greater than lam_bar, then skip the generation process
            #   and return None.
            if lam_value > lam_bar:
                print("intensity %f is greater than upper bound %f." % (lam_value, lam_bar), file=sys.stderr)
                return None
            # accept
            if lam_value >= D * lam_bar:
                # retained_points.append(homo_points[i])
                retained_points = np.concatenate([retained_points, homo_points[[i], :]], axis=0)
            # monitor the process of the generation
            if verbose and i != 0 and i % int(homo_points.shape[0] / 10) == 0:
                print("[%s] %d raw samples have been checked. %d samples have been retained." % \
                    (arrow.now(), i, retained_points.shape[0]), file=sys.stderr)
        # log the final results of the thinning algorithm
        if verbose:
            print("[%s] thining samples %s based on %s." % \
                (arrow.now(), retained_points.shape, self.lam), file=sys.stderr)
        return retained_points

    def generate(self, T=[0, 1], S=5, batch_size=10, min_n_points=5, verbose=True):
        """
        generate spatio-temporal points given lambda and kernel function
        """
        points_list = []
        sizes       = []
        max_len     = 0
        b           = 0
        
        # generate inhomogeneous poisson points iterately
        count_1 = 0
        count_2 = 0
        while b < batch_size:
            homo_points = self._homogeneous_poisson_sampling(T=T, S=S)
            points      = self._inhomogeneous_poisson_thinning(homo_points, verbose)
            # print(points)
            count_1 += 1
            if points is None or len(points) < min_n_points:
                continue
            count_2 += 1
            max_len = points.shape[0] if max_len < points.shape[0] else max_len
            points_list.append(points)
            sizes.append(len(points))
            print("[%s] %d-th sequence is generated." % (arrow.now(), b+1), file=sys.stderr)
            b += 1
        # fit the data into a tensor
        data = np.zeros((batch_size, max_len, 2))
        for b in range(batch_size):
            data[b, :points_list[b].shape[0]] = points_list[b]
        
        return data, sizes, count_2/count_1