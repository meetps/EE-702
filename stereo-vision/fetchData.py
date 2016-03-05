import urllib2
import bs4
import os

url = "http://vision.middlebury.edu/stereo/data/scenes2006/ThirdSize/zip-7views/"

request = urllib2.Request(url)
response = urllib2.urlopen(request)
soup = bs4.BeautifulSoup(response)

links = [] 

print("Downloading Begins | Get some coffee")

for a in soup.findAll('a'):
  if '7views.zip' in a['href']:
    links.append(a['href'])
    zipFile = urllib2.urlopen(url + a['href'])
    output = open(a['href'],'wb')
    output.write(zipFile.read())
    output.close()
    print("Downloaded " + a['href'])
  else:
    pass

os.system("unzip '*.zip'")
os.system("rm *.zip")
# print links
