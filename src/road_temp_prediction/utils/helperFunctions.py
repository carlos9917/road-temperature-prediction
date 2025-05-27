'''
Python version of the calculateShadows.sh bash script
originally used to call Grass

Nomenclature
Stretch: refers to a particular road. The original
data set was classified according to county, station number
and road section. Not sure if this naming is consistent

'''
from datetime import datetime
import argparse
from argparse import RawTextHelpFormatter
import configparser
from collections import OrderedDict
import subprocess
import pandas as pd
import logging
import numpy as np
import os
logger = logging.getLogger(__name__)

def read_station(station_details) -> pd.DataFrame:
    """
    Create dataframe for station with details provided
    station_details is a string with name, sensor1, sensor2,easting and northing
    separated by commas
    """
    data = pd.DataFrame(columns=['easting','norting','station','county','roadsection'])
    station = station_details.split(",")[0]
    east = station_details.split(",")[1]
    nort = station_details.split(",")[2]
    data["easting"] = [east]
    data["norting"] = [nort]
    data["station"] = [station]
    data["county"] = [0]
    data["roadsection"] = [0]
    stretch_tile = '_'.join([str(int(float(nort)/1000)),str(int(float(east)/1000))])
    #I add a new column wth the stretch file
    data['tile'] = [stretch_tile]
    return data

def read_stretch(stretchfile) -> pd.DataFrame:
    '''
    Read the stretch data, with format
    easting|norting|county|station|roadsection
    stretchlist.columns=['easting','norting','id1','station','id2']

    527087.842096|6250499.367625|33|137280|131
    '''
    data=pd.read_csv(stretchfile,sep='|',header=None,dtype=str)
    #data.columns=['easting','norting','county','station','roadsection'] #CHANGED ON 20210924
    data.columns=['easting','norting','station','county','roadsection'] #CHANGED ON 20210924
    stretch_tile=[]
    for k,nort in enumerate(data.norting):
        east=data.easting.values[k]
        #stretch_tile.append('_'.join([str(int(nort/1000)),str(int(east/1000))]))
        stretch_tile.append('_'.join([str(int(float(nort)/1000)),str(int(float(east)/1000))]))
    #I add a new column wth the stretch file    
    data['tile'] = stretch_tile
    #print("read_stretch: checking what the data is looking like here")
    #print(data)
    return data

def read_conf(cfile):
    '''
    Read the config file
    '''
    conf = configparser.RawConfigParser()
    conf.optionxform = str
    logger.info("Reading config file %s"%cfile)
    conf.read(cfile)
    # read all options here:
    secs = conf.sections()
    shadowPars=OrderedDict()
    for sec in secs:
        if sec == "SHADOWS":
            options = conf.options(sec)
            for param in options:
                paramValue=conf.get(sec,param)
                shadowPars[param] = paramValue
                #print("getting this value %s for %s"%(paramValue,param))
    return shadowPars


def calc_tiles(stretchlist):
    '''
    Split the list of stations in their corresponding tiles.
    output is an ordered dict with all the tiles needed
    (formerly the /tmp/horangle-NN/tile_Norting_Easting directory).

    Each key is of the form XXXX_YYY, where XXX is the Norting
    and YYY the Easting (both divided by 1000).
    Each key contains a list with all the stretches contained in that tile

    '''
    #Calculate number of lines in the file:
    print("calc_tiles: selecting here the tile list for")
    print(stretchlist)
    tiles_list=OrderedDict()
    tiles_list_long=OrderedDict()
    inserted_tiles=[]
    for k,stretch in stretchlist.iterrows():
        #Crude way to insert the station information
        #NOTE: the keys of the returned dictionary
        #will be based on the tiles only, hence
        #any repeated tiles for stations with different
        #sensors and slightly different coordinates
        #will be ignored below
        insert='|'.join([str(stretch['easting']),str(stretch['norting']),str(stretch['county']),str(stretch['station']),str(stretch['roadsection'])])
        stretch_east=float(stretchlist['easting'][k])
        stretch_nort=float(stretchlist['norting'][k])
        stretch_tile = str(int(stretch_nort/1000))+'_'+str(int(stretch_east/1000))
        try:
            if not isinstance(tiles_list[stretch_tile], list):
                pass
        except:
                tiles_list[stretch_tile] = []
                st_info = "_".join([str(stretch['county']),
                            str(stretch['station']),
                            str(stretch['roadsection'])])
                tiles_list_long[stretch_tile+"_"+st_info] = []
        print(f"Inserting {insert} into {stretch_tile}")       
        tiles_list[stretch_tile].append(insert)
    return tiles_list

def read_tif_list(tfile):
    tif_list=np.loadtxt(tfile,delimiter=' ',dtype=str)
    return tif_list


def loop_tilelist(list_tiles, tif_files,tif_dir):
    '''
    Calculates list of tif_files needed by the list of tiles
    Returns dataframe with necessary tiles and tif files
    '''
    tileside=1
    mindist=1
    maxdistance=1000
    dist=maxdistance / 1000
    tiles=OrderedDict()
    files=OrderedDict()
    ctiles_list=[]
    tiles_list=[]
    files_list=[]
    coords_list=[]
    for tkey in list_tiles.keys():
        #tiles_list=[]
        #files_list=[]
        east = int(tkey.split('_')[1])
        north = int(tkey.split('_')[0])
        tile_east = 1000 * ( east + tileside )
        tile_west = 1000 * east
        tile_north = 1000 *( north + tileside )
        tile_south = 1000 * north
        if (dist < 1 ):
            dist=mindist # was 10, then was set to 1
        domain_east = tile_west / 1000 + dist
        domain_west = tile_west / 1000 - dist
        domain_north = tile_south / 1000 + dist
        domain_south =  tile_south / 1000 - dist
        for tfile in tif_files:
            sw_corner_east = int(tfile.split('_')[3].replace('.tif',''))
            sw_corner_north = int(tfile.split('_')[2])
            if ( sw_corner_east <= domain_east and sw_corner_east >= domain_west and
                 sw_corner_north <= domain_north and sw_corner_north >= domain_south):
                #tiles_list.append('_'.join([str(sw_corner_north), str(sw_corner_east)]))
                #files_list.append(os.path.join(tif_dir,tfile))
                #ctiles_list.append(tkey)
                #coords_list.append(list_tiles[tkey][0]) # Why am I appending only the first coordinate only?? CHANGE: 20210924. Also 3 lines above
                #One tile might contain more than one station
                for coordinate in list_tiles[tkey]:
                    ctiles_list.append(tkey)
                    tiles_list.append('_'.join([str(sw_corner_north), str(sw_corner_east)]))
                    files_list.append(os.path.join(tif_dir,tfile))
                    coords_list.append(coordinate)
    #Make a dataframe containing arrays: ctiles_list, tiles_list, files_list, coords_list
    data = pd.DataFrame({'station_tile':ctiles_list,'surrounding_tile':tiles_list,
        'tif_file':files_list,'coords':coords_list})
    #print("loop_tilelist: before passing data to be used by calc_shadows")
    #print(data)
    return data
