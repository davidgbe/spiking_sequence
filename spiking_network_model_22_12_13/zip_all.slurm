#!/bin/bash

## Presets
#SBATCH --job-name=cleanup

#SBATCH -p compute-bigmem
#SBATCH -A labfairhall

#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks=2
#SBATCH --mem=10G

#SBATCH --chdir=.

## Module import
cd robustness
for f in *; do zip -r -qq $f.zip $f; rm -rf $f; done

## Exit
exit 0
