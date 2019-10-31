#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Load original image
im1 = Image.open('chess.png')
im2 = Image.open('newyork.jpg')
im3 = Image.open('jellyfish.jpg')


# Convert to grayscale
im1_gray = im1.convert('LA')
im2_gray = im2.convert('LA')
im3_gray = im3.convert('LA')



# Convert to normalized numpy array

# Image 1
im1mat = np.array(list(im1_gray.getdata(band=0)), float)/255.
im1mat.shape = (im1_gray.size[1], im1_gray.size[0])
im1mat = np.matrix(im1mat)

# Image 2
im2mat = np.array(list(im2_gray.getdata(band=0)), float)/255.
im2mat.shape = (im2_gray.size[1], im2_gray.size[0])
im2mat = np.matrix(im2mat)

# Image 3
im3mat = np.array(list(im3_gray.getdata(band=0)), float)/255.
im3mat.shape = (im3_gray.size[1], im3_gray.size[0])
im3mat = np.matrix(im3mat)






print "Compression:"

# Perform SVD decomposition

u1, s1, v1 = np.linalg.svd(im1mat)
u2, s2, v2 = np.linalg.svd(im2mat)
u3, s3, v3 = np.linalg.svd(im3mat)


print "plot singular values"

plt.figure()

plt.title('Singular values')

plt.plot(np.log10(s1))
plt.plot(np.log10(s2))
plt.plot(np.log10(s3))

plt.legend(['chessboard.png', 'newyork.jpg', 'jellyfish.jpg'])

plt.savefig('singular_values.png')

print "end plotting"

#plt.show()



for r in range(2, 50, 10):
	
	cmpim1 = np.matrix(u1[:, :r])*np.diag(s1[:r])*np.matrix(v1[:r,:])

	plt.imshow(cmpim1, cmap='gray')
	
	plt.title("Image after r=%s" % r)
	#plt.show()
	result = Image.fromarray((cmpim1).astype(np.uint8))
	plt.savefig("chess_r=%g.png" % r)
	
	


for r in range(2, 900, 20):
	
	cmpim2 = np.matrix(u2[:, :r])*np.diag(s2[:r])*np.matrix(v2[:r,:])

	plt.imshow(cmpim2, cmap='gray')
	
	plt.title("Image after r=%s" % r)
	#plt.show()
	result = Image.fromarray((cmpim2).astype(np.uint8))
	plt.savefig("newyork_r=%g.jpg" % r)



for r in range(2, 900, 20):
	
	cmpim3 = np.matrix(u3[:, :r])*np.diag(s3[:r])*np.matrix(v3[:r,:])

	plt.imshow(cmpim3, cmap='gray')
	
	plt.title("Image after r=%s" % r)
	#plt.show()
	result = Image.fromarray((cmpim3).astype(np.uint8))
	plt.savefig("jellyfish_r=%g.jpg" % r)







































































































"""

def Load_image(image):
	
	im_gray = Image.open(image).convert('LA')

	im_gray_mat = np.array(list(im_gray.getdata(band=0)), float)/255.
	
	im_gray_mat.shape = (im_gray.size[1], im_gray.size[0])

	im_gray = np.matrix(im_gray)

	plt.imshow(im_gray, cmap='gray')
	pt.show()
"""





