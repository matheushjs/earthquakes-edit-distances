

# Experiments for checking the effect of tlambda
for reg in ja nz gr jma; do
    for tlambda in 120 100 80 60 40 20 10 5 1 0.75 0.5 0.25 0.1 0.05; do
        python3 -u edit_distance_calc.py --region $reg --tlambda $tlambda --nthreads 24 --partial --partial-n 500 --outdir /media/mathjs/HD-ADU3/ --dry-run || true;
    done
done

# Experiments for checking effect of predict window size
for reg in ja nz gr jma; do
    for inputw in 3 2; do
        python3 -u edit_distance_calc.py --region $reg --tlambda 20 --inputw $inputw --nthreads 24 --partial --partial-n 500 --outdir /media/mathjs/HD-ADU3/ --dry-run || true;
    done
done

# Get full distance matrices for the subregions (inputw 21 outputw 3)
for reg in toho stil well jmatoho; do
    for tlambda in 100; do
        python3 -u edit_distance_calc.py --region $reg --tlambda $tlambda --nthreads 24 --inputw 21 --outputw 3 --outdir /media/mathjs/HD-ADU3/ --dry-run || true;
    done
done

# Get full distance matrices for the subregions (inputw 7 outputw 1)
for reg in toho stil well jmatoho; do
    for tlambda in 100; do
        python3 -u edit_distance_calc.py --region $reg --tlambda $tlambda --nthreads 24 --outdir /media/mathjs/HD-ADU3/ --dry-run || true;
    done
done