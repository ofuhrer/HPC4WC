#!/bin/sh

clear

# ----------------------------------------------------------------------- #
# PRECONDITION
# ensure that 'make_initial_fields.cpp' and '2d_Burger.cpp' are both compiled with these types:
# 'make_initial_fields_longdouble',   '2d_Burger_longdouble'  -> longdouble
# 'make_initial_fields_double',       '2d_Burger_double'      -> double
# 'make_initial_fields_single',       '2d_Burger_single'      -> float
# 'make_initial_fields_half',         '2d_Burger_half'        -> __fp16

# ----------------------------------------------------------------------- #
# VARIABLES
echo "Setting up..."

# Log file names
logfile_longdouble=log_longdouble.txt
logfile_double=log_double.txt
logfile_single=log_single.txt
logfile_half=log_half.txt

# type names
prec_longdouble=longdouble
prec_double=double
prec_single=single
prec_half=half

            # timestep = 1e-4 by default
nx=20      # spatial resolution in x direction
ny=20      # spatial resolution in y direction
nz=1        # spatial resolution in z direction
Tend=1      # end time of simulation
Tsave=1     # simulation time to be saved

# number of time steps to safe
nums=$((Tsave * 1000))

echo "Done"
echo ""
# ----------------------------------------------------------------------- #

# ----------------------------------------------------------------------- #
# CLEANING UP
echo "CLEANING UP..."
# rm input_data/*
# rm output_data/*
# rm animations/*
# >$logfile_longdouble
# >$logfile_double
# >$logfile_single
# >$logfile_half

echo "Done"
echo ""
# ----------------------------------------------------------------------- #

# ----------------------------------------------------------------------- #
# MAKE_INITIAL_FIELDS
echo "MAKE_INITIAL_FIELDS"
echo "Running 'make_initial_fields'..."
./make_initial_fields_longdouble --nx $nx --ny $ny --nz $nz
# ./make_initial_fields_double --nx $nx --ny $ny --nz $nz
# ./make_initial_fields_single --nx $nx --ny $ny --nz $nz
# ./make_initial_fields_half --nx $nx --ny $ny --nz $nz
echo "Done"
echo ""
# ----------------------------------------------------------------------- #

# ----------------------------------------------------------------------- #
# 2D_BURGER
echo "2D_BURGER"
echo "Running '2d_Burger' for 'longdouble'..."
./2d_Burger_longdouble --nx $((nx / 4)) --ny $((ny / 4)) --nz $nz --Tend $Tend --nums $nums  >> $logfile_longdouble
./2d_Burger_longdouble --nx $((nx / 2)) --ny $((ny / 2)) --nz $nz --Tend $Tend --nums $nums  >> $logfile_longdouble
./2d_Burger_longdouble --nx $nx --ny $ny --nz $nz --Tend $Tend --nums $nums  >> $logfile_longdouble
echo "Done"
echo ""


echo "Running '2d_Burger' for 'double'..."
./2d_Burger_double --nx $((nx / 4)) --ny $((ny / 4)) --nz $nz --Tend $Tend --nums $nums >> $logfile_double
./2d_Burger_double --nx $((nx / 2)) --ny $((ny / 2)) --nz $nz --Tend $Tend --nums $nums >> $logfile_double
./2d_Burger_double --nx $nx --ny $ny --nz $nz --Tend $Tend --nums $nums >> $logfile_double
echo "Done"
echo ""


echo "Running '2d_Burger' for 'single'..."
./2d_Burger_single --nx $((nx / 4)) --ny $((ny / 4)) --nz $nz --Tend $Tend --nums $nums >> $logfile_single
./2d_Burger_single --nx $((nx / 2)) --ny $((ny / 2)) --nz $nz --Tend $Tend --nums $nums >> $logfile_single
./2d_Burger_single --nx $nx --ny $ny --nz $nz --Tend $Tend --nums $nums >> $logfile_single
echo "Done"
echo ""


echo "Running '2d_Burger' for 'half'..."
./2d_Burger_half --nx $((nx / 4)) --ny $((ny / 4)) --nz $nz --Tend $Tend --nums $nums >> $logfile_half
./2d_Burger_half --nx $((nx / 2)) --ny $((ny / 2)) --nz $nz --Tend $Tend --nums $nums >> $logfile_half
./2d_Burger_half --nx $nx --ny $ny --nz $nz --Tend $Tend --nums $nums >> $logfile_half
echo "Done"
echo ""
# ----------------------------------------------------------------------- #

# ----------------------------------------------------------------------- #
# PYTHON VISUALISATION
echo "BURGER_ANIMATION"
echo "Running 'Burger_animation.py' for 'longdouble'..."
# python Burger_animation.py --nx $((nx / 4)) --ny $((ny / 4)) --nz $nz --Tend $Tend --prec $prec_longdouble --nums $nums
# python Burger_animation.py --nx $((nx / 2)) --ny $((ny / 2)) --nz $nz --Tend $Tend --prec $prec_longdouble --nums $nums
# python Burger_animation.py --nx $nx --ny $ny --nz $nz --Tend $Tend --prec $prec_longdouble --nums $nums
echo "Done"
echo ""

echo "Running 'Burger_animation.py' for 'double'..."
# python Burger_animation.py --nx $((nx / 4)) --ny $((ny / 4)) --nz $nz --Tend $Tend --prec $prec_double --nums $nums
# python Burger_animation.py --nx $((nx / 2)) --ny $((ny / 2)) --nz $nz --Tend $Tend --prec $prec_double --nums $nums
# python Burger_animation.py --nx $nx --ny $ny --nz $nz --Tend $Tend --prec $prec_double --nums $nums
echo "Done"
echo ""

 echo "Running 'Burger_animation.py' for 'single'..."
# python Burger_animation.py --nx $((nx / 4)) --ny $((ny / 4)) --nz $nz --Tend $Tend --prec $prec_single --nums $nums
# python Burger_animation.py --nx $((nx / 2)) --ny $((ny / 2)) --nz $nz --Tend $Tend --prec $prec_single --nums $nums
# python Burger_animation.py --nx $nx --ny $ny --nz $nz --Tend $Tend --prec $prec_single --nums $nums
echo "Done"
echo ""

echo "Running 'Burger_animation.py' for 'half'..."
# python Burger_animation.py --nx $((nx / 4)) --ny $((ny / 4)) --nz $nz --Tend $Tend --prec $prec_half --nums $nums
# python Burger_animation.py --nx $((nx / 2)) --ny $((ny / 2)) --nz $nz --Tend $Tend --prec $prec_half --nums $nums
# python Burger_animation.py --nx $nx --ny $ny --nz $nz --Tend $Tend --prec $prec_half --nums $nums
echo "Done"
echo ""

echo "Running visualisation..."
python visualisation.py --nx $nx --ny $ny --nz $nz --Tend $Tend --nums $nums --mode u
python visualisation.py --nx $nx --ny $ny --nz $nz --Tend $Tend --nums $nums --mode v
echo "Done"
echo ""

# ----------------------------------------------------------------------- #

echo "That's it. Yay!"
echo ""

exit 0