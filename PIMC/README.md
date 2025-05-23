# PIMC: Path Integral Monte Carlo Programs

This directory contains a few short Python programs and a Julia PIMC program.

# Main Program
- `PIMC_main.jl` is the main program for PIMC in the canonicals ensemble. 
 - Reads in the temperature as a command line parameter, for example\
   `julia PIMC_main.jl T=1.0`
   start a PIMC simulation at T=1 (units depend on the system, usually K)
# Modules
- `PIMC_Common.jl`
  - A few sets of parameters for the three test systems in the book
     - `case = 1` is two noninteracting bosons in a harmonic trap - exact solution is known.
	 - `case = 216` liquid He with 16 atoms 
	 - `case = 232` liquid He with 32 atoms 
	 - `case = 264` liquid He with 64 atoms 
	 - `case = 3` noninteragting bosons in 3D 
 - `PIMC_Systems` potential functions for systems mentioned above
 - `PIMC_Moves.jl` PIMC moves (updates): bisection, worm etc.
 - `PIMC_Structs.jl` most Julia structs and PIMC initialization (some already in `PIMC_main.jl`)
 - `PIMC_Measurements.jl` measurements done during PIMC simulation; what and how often is set in `PIMC_main.jl`
 - `PIMC_Reports.jl` what to print on screen or to the results HFD5 otput file
 - `PIMC_Primitive_Action.jl` and `PIMC_Chin_Action.jl` primitive action and Chin action, with specific energy estimators
 - `PIMC_Utilities.jl` short utility programs
 - `QMC_Statistics.jl` functions to collect samples and block data for error estimation (same as in DMC)
 
