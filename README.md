# voxid 
singing voice analysis and detection tools 
 
Usage: 
	import autotune 
	autotune_baseline_test() 
	autotune.rms_spectrum_test(song='tainted', tuning_f0=110.) # RMS power of equal-temperament bins  for song 'tainted' 
	autotune.mixture_test(song='tainted', tuning_f0=110.) # apply autotune algorithm to song/vocals .wav and mix with song/background.wav 
	autotune.predominant_melody_test(song='tained', tuning_f0=110.) 

Author: Michael A. Casey 
Copyright (C) 2015, Bregman Media Labs, Dartmouth College 
License: Apache 2.0, see LICENSE file 

