def f(x, a=-100, b=100):
    '''General mapping function
    Maps a sequence with maximum value b and 
    minimum value a between 0 and 1 '''
    
    return (x-a)/(b-a)
    
def x(f, a=-100, b=100):
    '''Inverse mapping function'''
    
    return f*(b-a) + a
