#!/bin/bash
for i in {3..5}
do
    for N in {5250000..50000000..250000}
    do
        for A in "SCD" "DM" "CM"
        do
            for D in "dense" #"fluid" "sparse"
            do
                CUDA_VISIBLE_DEVICES=1 ./particles -n $N -a $A -d $D &> result/$i-$N-$A-$D
            done
        done
    done
done
