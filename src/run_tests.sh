#!/bin/bash
for i in {1..2}
do
    for N in 100000 
    do
        for A in "DC" "SCD" "DM" "CM" "SAS"
        do
            for D in "dense" "fluid" "sparse"
            do
                ./particles -n $N -a $A -d $D > result/$i-$N-$A-$D
            done
        done
    done
done