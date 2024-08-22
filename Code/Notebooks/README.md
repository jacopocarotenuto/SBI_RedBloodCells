# Notebooks
Here you can find the Jupyer notebooks with the results of our analysis, organized by topics. 
1. ```OrnsteinUhlenbeck```: simulations, observables, and inference of the Ornstein-Uhlenbeck stochastic process. This file contains also the definition of the simulator and the SBI pipeline. </br>
2. ```RunTumble```: simulations, observables, and inference of the Run&Tumble stochastic process. It follows the very same schema of the ```OrnsteinUhlenbeck``` to make the comparison easier. </br>
3. ```BloodCells```: analysis of the Blood Cells model. This includes the SBI pipeline, analysis of the entropy production - and its inference -, and considerations about the general performance of the model. The code about the simulations and the summary statistics is instead in the ```InternalLibrary```. This notebook is not meant to be run completely each time.</br>
4. ```s_redx```: some tests and benchmarks about the summary statistics *s_redx* and *Cxx* for the Blood Cells model. </br>

The ```data``` subfolder contains some pickle files to be loaded directly into the notebooks. 