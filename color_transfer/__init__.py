# 导入numpy及opencv库
import numpy as np
import cv2

def color_transfer(source, target):
	"""
	
	使用LAB颜色空间的均值（Mean）和标准差（Standard Deviation），把source的颜色应用到target上。

	算法参考 《Color Transfer between Images》 by Reinhard et al., 2001. 论文实现

	Parameters:
	-------
	source: NumPy array
		OpenCV image in BGR color space (the source image)
	target: NumPy array
		OpenCV image in BGR color space (the target image)

	Returns:
	-------
	transfer: NumPy array
		OpenCV image (w, h, 3) NumPy array (uint8)
	"""
	# 利用opencv的方法，把图片的RGB颜色转化为LAB颜色。 (note: OpenCV
	# expects floats to be 32-bit, so use that instead of 64-bit)
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

	# 对source和target图片，分别计算LAB各个通道的均值、标准差。
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

	# 对target图片的LAB各通道减去对应的均值，cv2.split用于通道拆分
	(l, a, b) = cv2.split(target)
	l -= lMeanTar
	a -= aMeanTar
	b -= bMeanTar

	# 通过标准差来校对target图片的LAB通道。
	l = (lStdTar / lStdSrc) * l
	a = (aStdTar / aStdSrc) * a
	b = (bStdTar / bStdSrc) * b

	# 加上source图片的均值
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc

	# 把LAB各通道的值，限制在0至255之间，小于0的取0，大于255的则取255
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)

	# 把LAB通道合并，并转化为RGB值。cv2.merge通道合并
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
	
	# 返回颜色转化后的图片
	return transfer

def image_stats(image):
	"""
	Parameters:
	-------
	image: NumPy array
		OpenCV image in L*a*b* color space

	Returns:
	-------
	Tuple of mean and standard deviations for the L*, a*, and b*
	channels, respectively
	"""
	# 计算LAB各通道的均值与标准差
	(l, a, b) = cv2.split(image)
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())

	# return the color statistics
	return (lMean, lStd, aMean, aStd, bMean, bStd)
