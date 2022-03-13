#!/bin/bash

cwd="$(pwd)"
base=`basename "$cwd"`
cmdstan="$HOME/Documents/cmdstan-2.24.0/"
make "$cwd/$base" -C "$cmdstan"

w=200
N=2000

for i in 1 2 3
do
    ./$base sample num_samples=$N num_warmup=$w data file=${base}.data.json output file=${base}_${i}.csv &
done

