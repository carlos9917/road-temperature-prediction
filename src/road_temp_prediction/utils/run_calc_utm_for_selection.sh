#!/usr/bin/env bash

#this example assumes station data in "road_stretch" format (station,name,lat,lon)
INPUT=../data/selected_kriging_points.csv
python calcUTM.py -ifile $INPUT 
