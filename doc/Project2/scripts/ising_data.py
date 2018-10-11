import numpy as np

def ising_energies(states, L, J_const=1.0):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=J_const
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E
    
    
if __name__ == '__main__':
    #np.random.seed(12)
    
    # Simple example
    
    # define Ising model params
    L=5         # system size
    N=5     # Number of states

    # create random Ising states
    states=np.random.choice([-1, 1], size=(N,L))

    # calculate Ising energies
    energies=ising_energies(states,L)

    print(states)
    print(energies)

