# python 2.7
import numpy
import time
import urllib2
from bs4 import BeautifulSoup
latest = 463415
failed = list()
with open('results.txt', 'a') as file:
	for i in numpy.arange(latest, 0, -1);
	url = 'http://www.gunviolencearchive.org/incident/' + str(i)
	req = urllib2.Request(url,
	                      data = None,
	                      headers={
		                      'Accept': 'text/html,text/plain',
		                      'Accept-Language': 'en-US',
		                      'Connection': 'close',
		                      'Referer': 'https://google.com',
		                      'User-Agent': 'larryfenn@gmail.com - crawling your db one-time only'
		                      })
	try:
		resp = urllib2.urlopen(req)
		soup = BeautifulSoup(resp.read(), 'html.parser')
		result = soup.find(id='block-system-main')
		file.write(str(i) + str(result))
	except urllib2.HTTPError as e:
		if e.code != 404:
			failindices.append(i)
			print e.code
	time.sleep(.2)
f = open('failindices.txt', 'w')
f.write(failindices)
f.close()
