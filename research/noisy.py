from __future__ import division
import numpy as np
np.random.seed(2)
from scipy.stats import norm
from scipy.special import ndtri
from numpy import mean
from numpy import var
from math import sqrt
from math import log
from math import floor
from math import pow
from time import clock

def sample(logger, x, y):
	"""sample from the test function to be optimized on.
	Required: It needs to be unimodal in one axis, and monotone in the other.
	log - logging object
	"""
	z = 2*x*(x - 1) - y + 101 + norm.rvs(scale=.25)
	#z = 2*x*(x - 1) - y + 101 + norm.rvs(scale=y/100 + .1*abs(x))
	logger.logSample(x, y, z)
	return z

def discern(x1, x2, y, const, logger, findminFlag, alpha=.05, tol=.5, priorsamples=0, priorsampleflag=0, n0=2):
	"""Discern determines which of f(x_1) or f(x_2) has a smaller mean, given
	only noisy evaluations, with significance level 1 - alpha, indifference
	level tol, and constraint level const.

	Arguments:
	x1, x2 - the two probe arguments for fcn.
	n0 - initial sample count.
	alpha - significance level (1-alpha).
	tol - indifference level on confidence interval size.
	const - constraint, below which we terminate.
	log - logging object

	Returns:
	(0, samples) - tolerance level reached.
	(1, samples) - x1 is smaller.
	(2, samples) - x2 is smaller.
	3 - hard constraint achieved by x1.
	4 - hard constraint achieved by x2.
	"""
	samplecount = 0
	c = ndtri(1-alpha/2)
	x1samples = list()
	x2samples = list()
	if priorsampleflag == 1:
		x1samples = priorsamples
	if priorsampleflag == 2:
		x2samples = priorsamples
	# initial sampling and estimator construction and check
	for n in range(n0):
		if priorsampleflag != 1:
			x1samples.append(sample(logger, x1, y))
		if priorsampleflag != 2:
			x2samples.append(sample(logger, x2, y))
	n1 = len(x1samples)
	n2 = len(x2samples)
	x1var = var(x1samples, ddof=1)
	x2var = var(x2samples, ddof=1)
	# main loop body: construct mean and variance, then apply checks
	while max(sqrt(x1var/n1), sqrt(x2var/n2)) >= tol/c:
		x1mean = mean(x1samples)
		x2mean = mean(x2samples)
		x1var = var(x1samples, ddof=1)
		x2var = var(x2samples, ddof=1)
		if abs(x1mean - x2mean) >= c*(sqrt(x1var/n1) + sqrt(x2var/n2)):
			if x1mean <= x2mean:
				#logger.logEvent(1, "Chose x1") LOGGING EVENT
				return (1, x1samples)
			else:
				#logger.logEvent(2, "Chose x2") LOGGING EVENT
				return (2, x2samples)
		if ((x1mean + c*sqrt(x1var/n1)) < const) and not findminFlag:
			#logger.logEvent(3, "x1 satisfies constraint") LOGGING EVENT
			return (3,0)
		if ((x2mean + c*sqrt(x2var/n2)) < const) and not findminFlag:
			#logger.logEvent(4, "x2 satisfies constraint") LOGGING EVENT
			return (4,0)
		# if none of these checks pass, refinement
		if sqrt(x1var*n2*(n2+1))*(sqrt(n1+1) - sqrt(n1)) >= sqrt(x2var*n1*(n1+1))*(sqrt(n2+1)-sqrt(n2)):
			n1 += 1
			x1samples.append(sample(logger, x1, y))
		else:
			n2 += 1
			x2samples.append(sample(logger, x2, y))
	#logger.logEvent(0, "Can't tell; chose the lesser of x1, x2")LOGGING EVENT
	if (mean(x1samples) > mean(x2samples)):
		return (2, x2samples)
	else:
		return (1, x1samples)

def find_min_local(left, right, y, const, logger, htol, findminFlag, alpha=.05, tol=.5):
	"""Performs golden section search until one of the following happens:
	candidate interval sufficiently small, constraint satisfied.
	Arguments:
	left, right - bounds on candidate interval
	logger - logging object
	htol - horizontal tolerance

	Returns:
	(10, value) - tolerance level reached with value at midpoint of (left, right)
	(11, value) - hard constraint achieved by x1 probe
	(12, value) - hard constraint achieved by x2 probe
	"""
	gr = (sqrt(5) - 1)/2
	carrysamples = list()
	sampleflag = 0
	while (right - left) >= htol:
		x1 = left + (right - left)*(1 - gr)
		x2 = left + (right - left)*gr
		#print(left, right)
		#sampleflag = 0
		returncode = discern(x1, x2, y, const, logger, findminFlag, alpha, tol, carrysamples, sampleflag)
		#print(returncode[0])
		if (returncode[0] == 0):
			carrysamples = returncode[1]
			sampleflag = 2
			right = x2
		if (returncode[0] == 1):
			carrysamples = returncode[1]
			sampleflag = 2
			right = x2
		if (returncode[0] == 2):
			carrysamples = returncode[1]
			sampleflag = 1
			left = x1
		if (returncode[0] == 3):
			return (11, x1)
		if (returncode[0] == 4):
			return (12, x2)
	#print(left, right)
	#logger.logEvent(10, "GS min found") LOGGING EVENT
	return (10, (left+right)/2)

def robbins_munro(initial, y, logger, const, tol=.001, estsize=10):
	count = 1
	guess = initial
	while guess > 0:
		result = 0
		for i in range(estsize):
			result += sample(logger, guess, y)
		result = result / estsize
		step = .01*(result - const)
		if (result < const - tol):
			return (101, guess)
		guess = guess - step
		#print(guess)
		count += 1
	return (100, 0)

def kiefer_w(initial, y, logger, const, tol=.001):
	count = 20 #25 works
	guess = initial
	exp = 1
	while guess > 0:
		step = .05
		divisor = pow(count, -exp)
		lead = sample(logger, guess + divisor, y)
		tail = sample(logger, guess - divisor, y)
		result = (lead - tail)/(2*divisor)
		if (lead < const - tol):
			return (101, (guess + divisor))
		if (tail < const - tol):
			return (101, (guess - divisor))
		if (abs(result)) < .001:
			return (100, guess)
		guess = guess - step*result
		#print(guess)
		#count += 1
	return (100, 0)

def find_min_global(lower, upper, left, right, const, logger, alpha=.05, tol=.5):
	"""Performs bisection search based on the boolean of "satisfies constraint"
	Arguments:
	lower, upper - discrete domain bounds. lower -> failure guarantee, upper -> success
	left, right - continuous domain bounds
	const - constraint
	logger - logging object

	Returns:
	(x, y) - arguments of the supposed minimum of the function in question
	"""
	x = 0
	K = .25
	htol = sqrt(K**2 - (tol - K)**2)
	while upper > (lower + 1):
		#print("LOWER AND UPPER:", lower, upper)
		y = lower + floor((upper - lower)/2)
		gsout = find_min_local(left, right, y, const, logger, htol, False, alpha, tol)
		if gsout[0] == 10:
			#logger.logEvent(100, "GS min above constraint; bisect up") LOGGING EVENT
			lower = y
		else:
			x = gsout[1]
			#logger.logEvent(101, "GS met constraint; bisect down") LOGGING EVENT
			upper  = y
	return (find_min_local(left, right, upper, const, logger, htol, True, alpha, tol)[1], upper)

def find_min_global_rm(lower, upper, start, const, logger, tol=.00001, estsize=10):
	x = 0
	while upper > (lower + 1):
		y = lower + floor((upper - lower)/2)
		rmout = robbins_munro(start, y, logger, const, tol, estsize)
		if rmout[0] == 100:
			#logger.logEvent(100, "RM did not succeed; bisect up") LOGGING EVENT
			lower = y
		else:
			x = rmout[1]
			#logger.logEvent(101, "RM met constraint; bisect down") LOGGING EVENT
			upper = y
	return (x, upper)

def find_min_global_kw(lower, upper, start, const, logger, tol=.00001):
	x = 0
	while upper > (lower + 1):
		y = lower + floor((upper - lower)/2)
		kwout = kiefer_w(start, y, logger, const, tol)
		if kwout[0] == 100:
			#logger.logEvent(100, "KW did not succeed; bisect up") LOGGING EVENT
			lower = y
		else:
			x = kwout[1]
			#logger.logEvent(101, "KW met constraint; bisect down") LOGGING EVENT
			upper = y
	return (x, upper)

def trial(lower, upper, left, right, const, alpha, tol, count, logs):
	guesses = list()
	for i in range(count):
		logs.append(Logger())
		print("ON TRIAL NUMBER", i)
		guesses.append(find_min_global(lower, upper, left, right, const, logs[i], alpha, tol))
	return guesses

def trial_rm(lower, upper, start, const, tol, estsize, count, logs):
	guesses = list()
	for i in range(count):
		logs.append(Logger())
		print("ON TRIAL NUMBER", i)
		guesses.append(find_min_global_rm(lower, upper, start, const, logs[i], tol, estsize))
	return guesses

def trial_kw(lower, upper, start, const, tol, count, logs):
	guesses = list()
	for i in range(count):
		logs.append(Logger())
		print("ON TRIAL NUMBER", i)
		guesses.append(find_min_global_kw(lower, upper, start, const, logs[i], tol))
	return guesses

class Logger(object):
	"""Logger for use with optimizer. Stores the following data:
	Samples: a list of (float, float) 2-tuples.
	Events: a list of (int, String) indices and events.
	"""

	def __init__(self):
		self.samples = list()
		self.events = list()
		self.duration = 0

	def logSample(self, x, y, z):
		self.samples.append((x, y, z))
		self.duration += 1

	def logEvent(self, code, name):
		self.events.append((self.duration, code, name))


#print(discern(.3, .31, 20, 40, logger))
#print(find_min_local(.2, .8, 50, 0, logger, .1, .001, .1))
#print(find_min_global(1, 128, .2, .8, 20, logger, .01, .1))

logs = list()
simcount = 100

rmtol = .2
start = clock()
rmanswer = trial_rm(1, 128, 3, 60, rmtol, 1, simcount, logs)
end = clock()
rmans = open('rmans1.dat', 'w')
for i in range(simcount):
	rmans.write(str(logs[i].samples) + str("\n"))
rmans.write(str(end - start))
rmans.close()

logs = list()

kwtol = .2 #.2 works
start = clock()
kwanswer = trial_kw(1, 128, 3, 60, kwtol, simcount, logs)
end = clock()
kwans = open('kwans1.dat', 'w')
for i in range(simcount):
	kwans.write(str(logs[i].samples) + str("\n"))
kwans.write(str(end - start))
kwans.close()

logs = list()

gstol = .2
alpha = .0001
start = clock()
gsanswer = trial(1, 128, 0, 3, 60, alpha, gstol, simcount, logs)
end = clock()
gsans = open('gsans1.dat', 'w')
for i in range(simcount):
	gsans.write(str(logs[i].samples) + str("\n"))
gsans.write(str(end - start))
gsans.close()

#print(logs[0].samples)
#print(logs[0].events)
#print(answer)

#rmevents = open('rmevents.dat', 'w')
#rmevents.write(str(rmeventslist))


#kwevents = open('kwevents.dat', 'w')
#kwevents.write(str(kweventslist))


#gsevents = open('gsevents.dat', 'w')
#gsevents.write(str(gseventslist))

print(clock())

#for reference: the solution to the problem is intersection at y = 41, x = 0 or 1
# minimum at x = .5

#todo: work in K_y
# implement horizontal bounds as a dependent function of K_y, epsilon
# data processing
