using JSON3
using KomaMRI

include("src/ServerFunctions.jl")

json_string = read("/home/pablov/Downloads/Scanner.json", String)
json_scanner =JSON3.read(json_string)

sys = json_to_scanner(json_scanner)

json_string = read("/home/pablov/Downloads/Sequence.json", String)
json_seq =JSON3.read(json_string)
seq = json_to_seq(json_seq, sys)