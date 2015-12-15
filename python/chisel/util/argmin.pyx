import numpy as np
cimport numpy as np

cpdef int argmin(np.float_t[::1] L, np.float_t[::1] P, np.float_t[::1] Q):
    cdef int x = 0
    cdef int i
    if len(L) == 0:
        return -1
    for i in range(len(L)):
        if L[i] < L[x]:
            x = i
        elif L[x] == L[i]:
            if P[i] > P[x]:
                x = i
            elif P[i] == P[x]:
                if Q[i] > Q[x]:
                    x = i
    return x
