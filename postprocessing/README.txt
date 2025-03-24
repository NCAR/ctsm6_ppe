We like to use the casper batch system for postprocessing of the PPE. This folder contains some tools to enable that. Please note that these scripts should be run on CASPER and not derecho.

The idea is to make a python script ("foo.py") that is designed to work on one lhc member (e.g. lhc0000) and will write out one netcdf file, with whatever metrics and subsetting you are interested in. Consider perusing the existing .py files in this directory for examples, especially postp_lhc_basic.py. Beware that you need to generate dummy files for any simulations that don't have history output.

The python script should take two command line arguments: the member to work on and a directory to write to. Generally speaking we recommend writing to somewhere in your scratch space. You can then move the concatenated file to home, work, or campaign once you are satisfied with the results.

So the first step is to prototype the python script usually in a jupyter notebook. The second step is to transfer that to a .py file and try it on one lhc from the command line. (e.g. conda activate ppe-py \\ python postp_lhc_basic.py lhc0000 ./)

The third step is to edit the driver script that will apply it to all 1501 wave0 lhc members. Generally speaking, only the first 5 lines of driver.sh should need to be edited. If you don't have access to P93300041 you will need to edit template.sh with your project code. You may also need to substitute an appropriate conda environment in template.sh if you aren't set up with ppe-py.

The final step is to run your full postprocessing task. We recommend:
bash driver.sh &> driver.log &

You can monitor driver.log for progress (tail driver.log), which will report the cumulative number of netcdfs written each minute. Once all the output netcdfs are generated, a concatenated file will appear in a date-labeled directory under your OUTDIR, e.g. $OUTDIR/c250324/concat/foo_concat.nc. Generally speaking we find post-processing the ensemble in this way will take 10-45 minutes, depending on the exact postp task. 

If driver.sh hangs, check qstat -u $USER to see if you have any jobs running. If not, something failed and you'll have to debug your python script. Consider trying to run driver.sh as is before editing anything to make debugging easier! Note also that driver.sh will persist long after your casper jobs all die if something goes wrong. At which point killing the backgrounded process is advisable (e.g. kill $(jobs -p)).<
