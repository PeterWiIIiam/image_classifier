# large datasets
# don't fit in memory 
# can come with essential metadata in hdf5

# challengees
# archiving
# 	we need to be able to read the data in 10 years possibly on another platform 
# processing
# 	programs must be able to acess the data and metadata in a fast and straightforward way
# sharing
# 	platform independent and self-describing
# explorable


# what is HDF5
# containers for objects, but not many kind of objects
# widely deployed C library - used in Matlab, IDL and others 
# ecosystem of users and developers 


# what's inside an HDF5
# three type of a HDF5 file
# datasets - homogeneous arrays of data
# groups - contains holding datasets and groups - can be used to build a tree to organize files
# attributes: arbitrary metadata on groups and datasets

# capability of dataset
# partial I/O: read and write just what you want
# automatic type conversion: when you read from the file, you specify the type. HDF5 will convert it for you
# transparent compression: automatic compression if wanted
# parallel reads and writes with MPI 

# HDF5 with python
# 1 way is h5py 
# general-perpose iterface to HDF5
# low-level interfaces and high-level 

# 2 way is pytables
# database/table oriented system based on HDF5 

# demo
# hdf5 is like a dictionary
import numpy as np
import h5py 

data = np.arange(10)
f = h5py.File("demo.hdf5", 'w')

f['mydata'] = data

dset = f['mydata']

dset.shape
# (10,)

dset.dtype
# dtype(int)

dset[0:6:2]
# only read 0 2 4

dset[[1,2,6]]
# only read 1, 2, 6

aset.attrs
# attribute of hdf5 object

aset.attrs['sampling rate'] = 100e6
aset.attrs['pressure'] = 15
# addint attributes onto the objects

# get hdfview

f.close()
f = h5py.File("demo.hdf5")
dset = f['mydata']
dset.name
# u/mydata 
# all the objects have full path name

root = f['/']
# root group
f['/path/dataset'] = data
dset2 = f['/path/dataset']
dset2.name
# u/path/dataset

grp = f['/path']
'mydata' in f
#  true

# filling the dataset
# you can create a compression dataset
dset3 = f.create_dataset(name = 'BIG' , shape = (1000, 1000, 1000, 1000), dtype = 'f', compression = 'gzip')
dset3.shape
# (1000, 1000, 1000, 1000)

dset4 = f.create_dataset("smaller", (1000,1000), dtype = 'f4', compression = 'gzip')
dset[:] = 42
f.flush()
# filters: lzf



