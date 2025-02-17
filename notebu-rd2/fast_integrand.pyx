# Cython implementation (save as fast_integrand.pyx)
from libc.math cimport fabs

cdef public double integrand(int n, double *args) nogil:
    cdef double x = args[0]
    cdef double a = args[1]
    cdef double b = args[2]
    return x**2 + a*x + b  # Replace with actual integrand