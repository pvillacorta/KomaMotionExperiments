cd(@__DIR__)

using HDF5

fid = h5open("sample.h5", "w")

sample = create_group(fid, "sample")

PD  = 1f0
T1  = 850f-3
T2  = 5f-3 
T2s = 0f0
Δw  = 0f0

sample["data"]       = permutedims(  [PD  .* ones(Float32, (2069943, 1));;;
                                      T1  .* ones(Float32, (2069943, 1));;;
                                      T2  .* ones(Float32, (2069943, 1));;;
                                      T2s .* ones(Float32, (2069943, 1));;;
                                      Δw  .* ones(Float32, (2069943, 1))], (3,2,1)  )
sample["offset"]     = zeros(Float32, (1, 3))
sample["resolution"] = zeros(Float32, (1, 3))

close(fid)

