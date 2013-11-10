# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:23:06 2013

@author: steve
"""

import urllib
works_file = 'AllWorks.txt'
jrp_url = "http://jrp.ccarh.org/data?a=musicxml&f="


for work in open(works_file) :
    filename = work.split('-')[0]
    print filename
    
    file_url = jrp_url+filename

    urllib.urlretrieve( file_url, "xml/{}.xml".format(filename) ) # Download the files
