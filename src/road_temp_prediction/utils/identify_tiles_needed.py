'''
Quick and dirty height calculation based on the scripts
I am using for the shadow calculations
It expects to use the danish surface model 
'''
import sqlite3
from datetime import datetime
import argparse
from argparse import RawTextHelpFormatter
import configparser
from collections import OrderedDict
import os
import pandas as pd
import sys
import logging
import shutil
import subprocess
import numpy as np
import shutil
#where the tiles are located
import helperFunctions as sf
from TIF_files import TIF_files as TIF_files

TILESDIR="/data/users/cap/DSM_DK"
TILESDIR="/data/projects/glatmodel/DSM_DK"
ZIPDATA="zip_contents.json"
def write_station(height,station,ofile):
    with open(ofile,"a+") as f:
        f.write(f"{height} {station}\n")

def get_zipfile(filepath,localpath):
    """
    filepath
    """
    filedest = os.path.join(localpath,filepath.split("/")[-1])
    if not os.path.isfile(filedest):
        try:
            #print(f"cp {filepath}")
            shutil.copy2(filepath,localpath)
            return filepath
            #out=subprocess.check_output(cmd,stderr=subprocess.STDOUT,shell=True)
        except OSError as err:
        #except subprocess.CalledProcessError as err:
            print(f"Error finding {filepath} to {localpath}: {err}")
    else:
        print(f"{filepath} already copied to {filedest}")

def unzip_file(zip_file,tif_file,dest):

    if not os.path.isfile(tif_file):
        try:
            local=os.getcwd()
            os.chdir(dest)
            cmd=f"unzip {zip_file}"
            out=subprocess.check_output(cmd,stderr=subprocess.STDOUT,shell=True)
            os.chdir(local)
        except subprocess.CalledProcessError as err:
            print(f"Error unzipping {zip_file} to {dest}: {err}")

def get_height(coords,elevation) -> float:
    import rasterio
    #coords = ((147.363,-36.419), (147.361,-36.430))
    #elevation = 'srtm_66_20.tif'
    with rasterio.open(elevation) as src:
        row, col = src.index(coords[0], coords[1])
        dem_data = src.read(1).astype('float64')
        height = dem_data[row,col]
        print(f"Local height for the station with coordinates {coords}: {height}")
        return height

#logger = logging.getLogger(__name__)
def setup_logger(logFile,outScreen=False):
    '''
    Set up the logger output
    '''
    global logger
    global fmt
    global fname
    logger = logging.getLogger(__name__)
    fmt_debug = logging.Formatter('%(levelname)s:%(name)s %(message)s - %(asctime)s -  %(module)s.%(funcName)s:%(lineno)s')
    fmt_default = logging.Formatter('%(levelname)-8s:  %(asctime)s -- %(name)s: %(message)s')
    fmt = fmt_default
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fname = logFile
    fh = logging.FileHandler(fname, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    if outScreen:
        logger.addHandler(ch) # Turn on to also log to screen


def main(args):
    all_zip_files=[]
    stretchlist=args.stretch_list
    outfile = args.outfile
    tilesDir=TILESDIR
    #The output will be written in this directory
    out_dir="extracted_tiles"
    src_dir="." #where the scripts are
    #This is the directory where I will copy and unpack the zip files:
    tilesdir=os.path.join(out_dir)
    now=datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S')
    print("Starting on %s"%now)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print("Output directory %s already exists"%out_dir)
    logFile=args.log_file
    setup_logger(logFile,outScreen=False)
    logger.info("Starting height calculation")
    #PRE-PROCESSING
    if os.path.isfile(args.stretch_list):
        print(f"Using list of stations {args.stretch_list}")
        if not "utm" in args.stretch_list: print(f"WARNING: File must contain UTM coordinates!")
        logger.info("Reading data from %s"%stretchlist)    
        stretch_data = sf.read_stretch(stretchlist)
    elif args.station != None:
        stretch_data = sf.read_station(args.station)
    #locate the files I need
    import json
    with open(ZIPDATA,"r") as f:
        zipcontents = json.load(f)
    avail_tifs=TIF_files(zipcontents)

    if stretch_data.empty:
        print("Station list %s empty!"%strechlist)
        print("Stopping height calculation for this list")
        sys.exit()

    tiles_list = sf.calc_tiles(stretch_data)
    tif_files = np.array(avail_tifs.tiflist)
    tiles_needed = sf.loop_tilelist(tiles_list,tif_files,tilesdir)
    #I want only the tile containing the station
    tiles_selected= tiles_needed[tiles_needed["station_tile"] == tiles_needed["surrounding_tile"]]
    keep_tiles = [f for f in tiles_selected["tif_file"]]
    if tiles_selected.empty:
        print("No tiles found for station(s) provided!")
        sys.exit(1)

    #TODO: if the stretch list is more than one station 
    # do a loop here where tiles_selected is one of each in the list above
    allData=OrderedDict()
    for label in ["station","height"]:
        allData[label] = []
    for k in range(len(tiles_selected)):
        this_tile=tiles_selected.iloc[k]
        #lookup_tifs = [this_tile["tif_file"].values[0].split("/")[-1]]
        lookup_tifs = [this_tile["tif_file"].split("/")[-1]]
        zipfile = avail_tifs.find_zipfiles(lookup_tifs)
        #the original function expects a list of files, but here I only need one
        zipfile = "".join(zipfile)
        localfile = os.path.join(TILESDIR,zipfile)
        zipfile_found = get_zipfile(localfile,out_dir)
        all_zip_files.append(zipfile_found)
        #unzip_file(zipfile,this_tile["tif_file"].values[0],out_dir)
        #elevation = this_tile["tif_file"].values[0]
        unzip_file(zipfile,this_tile["tif_file"],out_dir)
        elevation = this_tile["tif_file"]
        coords = (float(this_tile.coords.split("|")[0]),
                  float(this_tile.coords.split("|")[1]))

        stdata= this_tile.coords.split("|")
        station_id = stdata[3]+stdata[2].zfill(2)+stdata[4].zfill(2)
        #height = get_height(coords,elevation)
        #allData["station"].append(station_id)
        #allData["height"].append(height)
        #write_station(height,station_id,"processed_stations.txt")
        #Clean up
        print(f"Removing zip files in {out_dir}")
        delete_files = [os.path.join(out_dir,f) for f in os.listdir(out_dir) if "zip" in f]
        for f in delete_files:
            os.remove(f)
        print(f"Removing md5 files in {out_dir}")
        delete_files = [os.path.join(out_dir,f) for f in os.listdir(out_dir) if "md5" in f]
        for f in delete_files:
            os.remove(f)
        #print(f"Removing unnecessary files in {out_dir}")
        #delete_files = [os.path.join(out_dir,f) for f in os.listdir(out_dir) if f not in keep_tiles]
        #for f in delete_files:
        #    os.remove(f)
        #shutil.rmtree(out_dir)
    #os.rmdir(out_dir)
    #df_write = pd.DataFrame(allData)
    #df_write.to_csv(outfile,index=False)
    print(all_zip_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
             Example usage: ./get_height.py -sl coords_utm.csv''', formatter_class=RawTextHelpFormatter)

    parser.add_argument('-sl','--stretch_list',
           metavar='list of stations to be processed',
           type=str,
           default="./stretchlist_utm.csv",
           required=False)

    parser.add_argument('-st','--station',
           metavar='Station name and UTM coordinates, separated by commas (ie, 3011,519379.289172,6123006.457084)',
           type=str,
           default=None,
           required=False)

    parser.add_argument('-out','--outfile',
           metavar='Output file (ie, all_heights.csv)',
           type=str,
           default=None,
           required=True)

    parser.add_argument('-lg','--log_file',metavar='Log file name',
                                type=str, default='heights.log', required=False)

    args = parser.parse_args()
    print(args)
    main(args)
