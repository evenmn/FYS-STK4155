import numpy as np

def generate_J(L):
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    return J
    
    
def ising_energies(states,J):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
        
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E
    
    
def P_up(T):
    '''Probability of spin up'''
    
    if T < 1e-10:
        return 0
        
    else:
        return 1/(1+np.exp(2/T))
    
    
def P_dn(T):
    '''Probability of spin down'''
    
    if T < 1e-10:
        return 1
        
    else:
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
    energies = ising_energies(states, L)

    print(states)
    print(energies)
