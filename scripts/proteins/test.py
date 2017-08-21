from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy

if rank == 0:
   for i in range(1,comm.Get_size()):
       comm.send(i, dest=i)
   numpy.save('/scratch/scarste/test.npy', numpy.array([1234]))   
else:
   data = comm.recv(source=0)
   numpy.save('/scratch/scarste/test' + str(rank) + '.npy', numpy.array([1234]))
