#!/usr/bin/env pyhton


import numpy as np
import matplotlib.pyplot as plt
import time, os, random
import scipy.linalg 
import sys
import numpy.matlib


def solver_QR(A, b):
	

	t0 = time.clock()

	Q, R = np.linalg.qr(A)
	
	QT = np.transpose(Q)
	y = np.dot(QT, b)

	coefficients = back_substitution(R, y)


	t1 = time.clock()
	CPU_time = t1 - t0
	
	return coefficients, CPU_time


def Vandermonde_matrix(m, data_set):

	if data_set == 1:
		x, y = data_sets(data_set=1)

	elif data_set == 2:
		x, y = data_sets(data_set=2)

	else:
		os.system('clear')
		print "invalid data set"
		sys.exit()	

	n = y.shape[0]
	A = np.zeros((n, m))
	A[:,0] = 1.0
	
	for i in range(1, m):
		A[:,i] = x**i

	return A



def polynomial(x, coefficients, m):
	
	n = len(x)
	poly = np.zeros(n)
	
	for i in range(n):
		for j in range(m):
			poly[i] += coefficients[j]*x[i]**j

	return poly




def back_substitution(U, y):

	
	m = U.shape[0]
	coefficients = np.zeros(m)
	
	for i in reversed(range(0, m)):
		sum = 0

		for j in range(i+1, m):
			sum += U[i,j]*coefficients[j]

		coefficients[i] = (y[i] - sum)/float(U[i,i])

	return coefficients



def data_sets(data_set):
	
	n = 30
	start = -2
	stop = 2
	x = np.linspace(start, stop, n)
	eps = 2
	random.seed(2)
	r = np.random.randint(1, n)*eps

	if data_set == 1:
		y = x*(np.cos(r+0.5*x**3) + np.sin(0.5*x**3))

	elif data_set == 2:
		y = 4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r
	
	else:
		os.system('clear')
		print "data_set=%g is not a valid data set" % data_set		
		sys.exit()	
		
		
	return x, y





def cholesky_decomposition(A):

	
	B = np.dot(A.T, A)

	Ak = np.copy(B)
	
	L_ = np.zeros(Ak.shape)
	D = np.zeros(Ak.shape)

	for k in range(Ak.shape[1]):

		L_[:,k] = Ak[:,k]/float(Ak[k,k])
		
		
		D[k,k] = Ak[k,k]
		
		Ak = Ak - D[k,k]*np.c_[L_[:,k]].dot(np.c_[L_[:,k]].T)

	
	L = np.dot(L_, np.sqrt(D))
	LT = np.dot(np.sqrt(D), L_.T)
	
	return L, LT
	


def forward_substitution(L, b):
	

	m = L.shape[0]

	y_ = np.zeros(m)	

	for i in range(m):
		Sum = 0
		for j in range(i):
			Sum += L[i,j]*y_[j]

		y_[i] = (b[i] - Sum)/float(L[i,i])

	return y_


def solver_cholesky(A, b):

	t0 = time.clock()

	L, LT = cholesky_decomposition(A)
	
	# Solve Ly = A^T*b
	y_ = forward_substitution(L, np.dot(A.T, b))
	
	# Solve L^T*x = y_
	coefficients = back_substitution(L.T, y_)
		
	t1 = time.clock()
	CPU_time = t1 - t0

	return coefficients, CPU_time









if __name__=='__main__':
	
	# Create table of CPU-time measurements
	
	#os.mknod("CPU_time.txt")

	for data in 1, 2:
		for m in 3, 8:
			x, y = data_sets(data_set=data)
			A = Vandermonde_matrix(m=m, data_set=data)
			

			coeff_QR, CPU_QR = solver_QR(A, y)
			poly_QR = polynomial(x=x, coefficients=coeff_QR, m=m)

			
			#print 'cpu_qr', CPU_QR, 'data:', data, 'm:', m



			
			coeff_cholesky, CPU_cholesky = solver_cholesky(A, y)
			poly_cholesky = polynomial(x=x, coefficients=coeff_cholesky, m=m)



			#print 'cpu_chol', CPU_cholesky, 'data:', data, 'm:', m


			
			plt.figure()
			plt.plot(x, y, 'o')
			plt.plot(x, poly_QR)
			plt.plot(x, poly_cholesky, '--')
			plt.legend(['data', 'QR', 'Cholesky'])
			plt.title('Data vs Least Squares Fit, data set=%g, m=%g' % (data, m))
			plt.xlabel('X'); plt.ylabel('Y')

			plt.savefig('plot_data_%g_m=%g.pdf' % (data, m))
	
	

			#plt.show()
			
	
			
		














