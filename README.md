# Simulation Based Inference
## LCPb Final Project
This repository contains the materials about the final project of the "Laboratory of Computational Physics (MOD. B)" course in the "Physics of Data" master program, University of Padova. </br>
The project aims to explore the Simulation Based Inference to study three stochastic models: Ornstein-Uhlenbeck, Run&Tumble, and a dynamical model of the Blood Cells motion. SBI is a powerful and promising tool to infere the posterior pdf of some parameters directly from the simulations (or from some summary statistics of them), in a likelihood-free context. 

Group members: [Jacopo Carotenuto](https://github.com/jacopocarotenuto), [Paolo Lapo Cerni](https://github.com/paololapo), [Lorenzo Vigorelli](https://github.com/LorenzoVigorelli), [Arman Singh Bains](https://github.com/T3X3K) </br>
Supervisors: Prof. Marco Baiesi, Dr. Ivan Di Terlizzi

## About this repository
This GitHub repo is organized into two (plus one hidden) folders. Each folder has its own ```README``` with a brief description of the content of the files in it. The main folders are: </br>
1. ```Code``` contains both the Python pipelines and the analysis of the work done. You can find the results reached for each model and how we obtained them. </br>
2. ```InternalLibrary``` is the custom-made library we developed to make the pipelines coherent, optimized, and easier to read. This is mostly useful for the Blood Cells model, but some helper functions could be used also in other contexts. </br>
3. ```Data``` (hidden by the ```.gitignore```) is supposed to be organized into two subfolders: ```Simulations``` and ```SummaryStatistics```, containing the data about the Blood Cells model. Typically, we organized our data in batches (files .pkl) of 200 simulations each, divided by the day we did the simulations. The pipeline produces also a file ```done.txt``` with the list of the simulations already processed to obtain the set of summary statistics. </br>

## References
[1] Di Terlizzi, I., et al. "Variance sum rule for entropy production." *Science* 383.6686 (2024): 971-976. </br>
[2] Cranmer, Kyle, Johann Brehmer, and Gilles Louppe. "The frontier of simulation-based inference." *Proceedings of the National Academy of Sciences* 117.48 (2020): 30055-30062. </br>
[3] Tucci, Gennaro, et al. "Modeling active non-Markovian oscillations." *Physical Review Letters* 129.3 (2022): 030603. </br>
[4] Garcia-Millan, Rosalba, and Gunnar Pruessner. "Run-and-tumble motion in a harmonic potential: field theory and entropy production." *Journal of Statistical Mechanics: Theory and Experiment* 2021.6 (2021): 063203. </br>
[5] Papamakarios, George, and Iain Murray. "Fast ε-free inference of simulation models with bayesian conditional density estimation." *Advances in neural information processing systems* 29 (2016). </br>
[6] Greenberg, David, Marcel Nonnenmacher, and Jakob Macke. "Automatic posterior transformation for likelihood-free inference." *International Conference on Machine Learning*. PMLR, 2019. </br>
[7] Deistler, Michael, Pedro J. Goncalves, and Jakob H. Macke. "Truncated proposals for scalable and hassle-free simulation-based inference." *Advances in Neural Information Processing Systems* 35 (2022): 23135-23149. </br>
