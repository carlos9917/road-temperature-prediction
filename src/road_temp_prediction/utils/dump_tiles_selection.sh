#!/usr/bin/env bash
# Examples to run the processing
i=137
#for ST in $(awk -v count=$i -F "|" 'NR%10==0 {print $3","$1","$2}' ../latlon_UTM/vejvejr_stations_utm.csv > tmp_list_${i}.csv); do
CSV=/media/cap/extra_work/road_model/ml_pp/python/spatial_analysis/selected_stations_utm.csv
CSV=/media/cap/extra_work/road_model/ml_pp/python/spatial_prediction/selected_kriging_points_utm.csv
#split files
#split -d -a 4 -l 10 $CSV vv --additional-suffix ".csv"
#for F in vv*.csv; do
echo "Doing file $i: $F"
python ./identify_tiles_needed.py -sl $CSV -out nh_$i.csv

