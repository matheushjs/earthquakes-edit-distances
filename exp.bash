
#OUTDIR=/media/mathjs/HD-ADU3/
OUTDIR=./

# Experiments for checking the effect of tlambda
for reg in ja nz gr jma; do
    for tlambda in 0.025 0.01 0.0075 0.005; do
        python3 -u edit_distance_calc.py --region $reg --tlambda $tlambda --nthreads 23 --partial --partial-n 500 --outdir $OUTDIR || true;
    done
done

# Experiments for checking effect of predict window size
for reg in ja nz gr jma; do
    for inputw in 1; do
        python3 -u edit_distance_calc.py --region $reg --tlambda 20 --inputw $inputw --nthreads 23 --partial --partial-n 500 --outdir $OUTDIR || true;
    done
done

# Experiments for checking effect of predict window size
for reg in ja nz gr jma; do
    for inputw in 3 2 1; do
        python3 -u edit_distance_calc.py --region $reg --tlambda 1 --inputw $inputw --nthreads 23 --partial --partial-n 500 --outdir $OUTDIR || true;
    done
done

# Get full distance matrices for the subregions (inputw 21 outputw 3)
for reg in jmatoho; do
    for tlambda in 100; do
        python3 -u edit_distance_calc.py --region $reg --tlambda $tlambda --nthreads 23 --inputw 21 --outputw 3 --outdir $OUTDIR || true;
    done
done

# Get full distance matrices for the subregions (inputw 21 outputw 3)
for reg in toho stil well jmatoho; do
    for tlambda in 1; do
        python3 -u edit_distance_calc.py --region $reg --tlambda $tlambda --nthreads 23 --inputw 21 --outputw 3 --outdir $OUTDIR || true;
    done
done

# Get full distance matrices for the subregions (inputw 7 outputw 1)
for reg in toho stil well jmatoho; do
    for tlambda in 1 100; do
        python3 -u edit_distance_calc.py --region $reg --tlambda $tlambda --nthreads 23 --outdir $OUTDIR || true;
    done
done