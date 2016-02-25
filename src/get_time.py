#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from os.path import join

parser = argparse.ArgumentParser(description='Get files information')
parser.add_argument('dirname', metavar='dirname', nargs=1)

args = parser.parse_args()

for root, dirs, files in os.walk(args.dirname[0]):
	for f in files:
		file_name = join(root, f)
		
		counter = 0
		N = 0
		alg = None
		time = 0
		with open(file_name, 'r') as the_file:			
			for line in the_file:
				if counter == 1:
					N = int(line.split()[2])
				if counter == 2:
					alg = line.split(':')[1].strip()
				if counter == 6:
					time = float(line.split("=")[1].split()[0])
				counter += 1
		if time > 0:
			print alg, N, time
