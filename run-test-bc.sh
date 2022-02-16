#!/bin/bash

BC=nxbc.py

OPTCOM="--fwhm 0.05 -r 2 -s 1000 --thr 1e-4"

n3_50="-d 50"
n3_75="-d 75"
n3_150="-d 150"
n4="--N4"
n4_l3="$n4 -l3"
n4_l4="$n4 -l4"
n4_l5="$n4 -l5"

nomask=""
nomaskp="-p"

for IM in `pwd`/source/????.nii.gz ; do
  mask="-m ${IM%.nii.gz}_mask.nii.gz"
  for T in n3_50 n4_l4 ; do
    for M in mask nomask nomaskp ; do
      OUT=`pwd`/bc-out/nm-$T-$M/$(basename $IM .nii.gz)
      [ -d $(dirname $OUT) ] || mkdir -p $(dirname $OUT)
      DONE=$OUT-done
      [ -f $DONE ] && continue
      CMD="$BC $OPTCOM ${!T} ${!M} -i $IM -o $OUT.nii.gz"
      echo "$CMD" > $OUT.cmd
      #. $OUT.cmd 2>&1|tee $OUT-out.txt
      #touch $DONE
    done
  done
done

