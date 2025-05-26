#calculate UTM from lat lon
from pyproj import Proj
import pyproj
import sys
import pandas as pd
from collections import OrderedDict
import numpy as np
import os

def read_data_noshadow(ifile):
    '''
    Read the data from a file in the new format. Ex:
    4031,"Aarhus-Nord",0,0,0,10.111693,56.21921
    The first number is still station, while
    the 3 numbers after location are sensor numbers
    NOTE: I will just ignore the sensor3 in the output
    so it will produce a file looking the same as the
    one for the road stretches
    '''
    data=pd.read_csv(ifile,sep=",",header=None)
    data.columns=['station','location','sensor1','sensor2','sensor3','lon','lat']
    return data

def read_data_road_stretch(ifile):
    #Read the data from a file
    #5160,"Gjørup",9.36827374,56.65630341
    data=pd.read_csv(ifile,sep=",",header=None)
    data.columns=['station','location','lon','lat']
    return data

def latlon2utm(lat,lon):
    #this used to work in py36
    #etrs89=Proj("+init=EPSG:4258") 
    #UTM32N=pyproj.Proj("+init=EPSG:25832") 
    #old style:
    #east,north= pyproj.transform(etrs89,UTM32N,lon,lat)
    #print("{:.6f} {:.6f}".format(east,north))
    #modification for py38 version
    from pyproj import Transformer
    transformer=Transformer.from_crs(4258,25832,always_xy=True)
    for pt in transformer.itransform([(float(lon),float(lat))]): 
        res='{:.6f} {:.6f}'.format(*pt)
    east,north=res.split()
    return east,north

def calc_UTM_file(ifile,input_format="road_stretch"):
    '''
    Calculate the UTM coordinates for all stations in ifile
    and write the output
    if input_format is road stretch it will use county and roadsection
    '''
    #This is for the old format:
    if input_format == "road_stretch":
        data=read_data_road_stretch(ifile)
        dout=OrderedDict()
        dout['easting']=np.array([])
        dout['norting']=np.array([])
        dout['station']=np.array([])
        dout['county']=np.array([])
        dout['roadsection']=np.array([])
    elif input_format == "noshadow":
        data=read_data_noshadow(ifile)
        dout=OrderedDict()
        dout['easting']=np.array([])
        dout['norting']=np.array([])
        dout['station']=np.array([])
        dout['sensor1']=np.array([])
        dout['sensor2']=np.array([])
        #dout['sensor3']=np.array([]) #Ignoring this one for the moment. Not important
    else:    
        print(f"Input format {input_format} unknown! Stopping here")
        sys.exit(1)

    for k,lat in enumerate(data.lat.values):
        east,nort=latlon2utm(lat,data.lon.values[k])
        dout['easting'] = np.append(dout['easting'],east)
        dout['norting'] = np.append(dout['norting'],nort)
        dout['station']=np.append(dout['station'],str(data['station'].values[k]))
        if input_format == "road_stretch":
            #now I set these to 0 for the moment
            dout['roadsection']=np.append(dout['roadsection'],'0')
            dout['county']    =np.append(dout['county'],'0')
        else:
            dout['sensor1']=np.append(dout['sensor1'],str(data['sensor1'].values[k]))
            dout['sensor2']=np.append(dout['sensor2'],str(data['sensor2'].values[k]))
            #dout['sensor3']=np.append(dout['sensor3'],str(data['sensor3'].values[k]))
    if input_format == "road_stretch":
        write_out=pd.DataFrame({'easting':dout['easting'],'norting':dout['norting'],'station':dout['station'],'roadsection':dout['roadsection'],'county':dout['county']})
    else:
        write_out=pd.DataFrame({'easting':dout['easting'],'norting':dout['norting'],'station':dout['station'],'sensor1':dout['sensor1'],'sensor2':dout['sensor2']})
    ofile=os.path.split(ifile)[-1].replace(".csv","_utm.csv")
    print("output file %s"%ofile)
    write_out.to_csv(ofile,sep='|',float_format='%.3f',index=False,header=False)

if __name__ == '__main__':
    import argparse
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(description='''
            Convert lat/lon to UTM
            Give coordinates or use file
            Example usage: python3 -coords lat,lon
                           python3 -ifile file
                           ''',formatter_class=RawTextHelpFormatter)

    group1 = parser.add_argument_group("Command line")
    group2 = parser.add_argument_group("Input file")

    group1.add_argument('-coords',help='lat lon coordinates, separated by a comma',type=str,required=False)
                         
    group2.add_argument('-ifile',help='File with lat lon coordinates',type=str,required=False)
    group2.add_argument('-input_format',help='Format of input file (road_stretch or noshadow). Default is road_stretch',type=str,required=False,default="road_stretch")
    args = parser.parse_args()
    if args.coords != None:
        coords = args.coords.split(",")
        print(f"Using lat,lon coordinates {coords}")
        lat,lon = coords[0],coords[1]
        east,nort=latlon2utm(lat,lon)
        #print(str(east)+" "+str(nort))
        print(f"Easting and Norting: {east}, {nort}")
    elif args.ifile != None:    
        ifile = args.ifile
        input_format = args.input_format
        print(f"Using file {ifile} with format {input_format}")
        calc_UTM_file(ifile,input_format)
    else:
        print("No arguments given")
        sys.exit(1)
