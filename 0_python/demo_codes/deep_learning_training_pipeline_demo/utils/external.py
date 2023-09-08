import sys
import os

# get python version info.
PY2 = sys.version_info < (3,)

def download(url,dest = None):
	import tarfile
	if PY2:
		from urllib import  urlretrieve
	else:
		from urllib.request import urlretrieve

	if dest is None:
		dest = os.getcwd()

	filepath = os.path.join(dest,_filename(url))
	urlretrieve(url,filepath)

	with tarfile.open(filepath,'r:gz') as f:
		f.extractall(dest)
		f.close()

def _filename(url):
	if PY2:
		from urlparse import urlparse
	else:
		from urllib.parse import urlparse

	a = urlparse(url)

	return os.path.basename(a.path)

