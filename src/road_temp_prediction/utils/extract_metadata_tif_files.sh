#!/bin/bash
# Script to extract DSM metadata and format it for Python
# Extracts data from tif files present under extracted_files

DUMP=selected_tif_metadata.txt
echo "dsm_files = [" > $DUMP
LOCAL_PATH=$PWD

for file in $LOCAL_PATH/extracted_tiles/*.tif; do
  #echo "Processing $file..."
  
  # Extract size
  size=$(gdalinfo $file | grep "Size is" | sed 's/Size is //')
  width=$(echo $size | cut -d',' -f1)
  height=$(echo $size | cut -d',' -f2)
  
  # Extract origin
  origin=$(gdalinfo $file | grep "Origin =" | sed 's/Origin = //' | sed 's/,/ /')
  origin_x=$(echo $origin | cut -d' ' -f1)
  origin_y=$(echo $origin | cut -d' ' -f2)
  
  # Extract pixel size
  pixel_size=$(gdalinfo $file | grep "Pixel Size =" | sed 's/Pixel Size = (//' | sed 's/,/ /')
  pixel_width=$(echo $pixel_size | cut -d' ' -f1)
  
  # Extract corner coordinates
  upper_left=$(gdalinfo $file | grep "Upper Left" | sed 's/Upper Left  (//' | sed 's/).*//')
  lower_right=$(gdalinfo $file | grep "Lower Right" | sed 's/Lower Right (//' | sed 's/).*//')
  
  ul_x=$(echo $upper_left | cut -d',' -f1)
  ul_y=$(echo $upper_left | cut -d',' -f2)
  lr_x=$(echo $lower_right | cut -d',' -f1)
  lr_y=$(echo $lower_right | cut -d',' -f2)
  
  # Output formatted Python dictionary
  echo "    {" >> $DUMP
  echo "        'name': '$file'," >> $DUMP
  echo "        'origin': ($origin_x, $origin_y)," >> $DUMP
  echo "        'pixel_size': $pixel_width," >> $DUMP
  echo "        'size': ($width, $height)," >> $DUMP
  echo "        'extent': [$ul_x, $lr_y, $lr_x, $ul_y]" >> $DUMP
  echo "    }," >> $DUMP
done

echo "]" >> $DUMP
