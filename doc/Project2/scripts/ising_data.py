import numpy as np

def ising_energies(states, J_const=1.0):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    
    X = np.multiply(states[:,1:], states[:,:-1])
    J = np.full(X.shape[1], J_const)

    return X.dot(J)
    
    
def P_up(T):
    '''Probability of spin down'''
    return 1/(1+np.exp(2/T))
    
    
def P_dn(T):
    '''Probability of spin up'''
    return 1/(1+np.exp(-2/T))
    
    
def produce_states(size, T):
    '''Producing Ising states
    
    Arguments:
    ----------
    
    size:   List/scalar.
            Size of system.
            
    T:      Scalar.
            Dimensionless temperature
            T'=kT/E. '''
            
    return np.random.choice([1,-1], size, p=[P_up(T), P_dn(T)])
    
    
if __name__ == '__main__':
    #np.random.seed(12)
    
    # Simple example
    
    L=4     # System size
    N=4     # Number of states

    states = produce_states([L, N], T=100)

    # calculate Ising energies
    energies = ising_energies(states)

    print(states)
    print(energies)
