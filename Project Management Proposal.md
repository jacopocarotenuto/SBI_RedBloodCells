# SBI Inference

## The Pipeline

The proposed pipeline is composed of the following steps:
1. Parameters are sampled from the *prior distribution*.
2. The simulator is run with the sampled parameters. We obtain the simulated data with it's corresponding parameters. We calculate the *summary statistics* of the simulated data.
3. The SBI object is defined (SNPE, etc).
4. The SBI object is *trained* with summary statistics. A density estimator is obtained.
5. The density estimator is used to approximate the posterior distribution.
6. The posterior distribution is used to infer the parameter given the observed data.


### Parameters of the pipeline

- **Prior Distribution**: The prior distribution is the distribution from which the parameters are sampled. It is the initial belief about the parameters.
- **Simulator**: The simulator is the model that generates the data given the parameters. It is the model that we want to infer.
- **Summary Statistics**: The summary statistics are the features of the data that are used to train the SBI object. They are the features that are used to approximate the posterior distribution.
- **Training**: The training process can be done in different ways and with different parameters eg. smaller batch size, multi-round inference...

## Our Project Goal

Build a pipeline with the correct parameters for the task at hand.

High-level tasks:
[ ] Theoretical understanding of the problem
[ ] Implement the pipeline in a robust way
[ ] From the theory get best prior distribution (or create a test to find it)
[ ] Create the simulator *PERFORMANCE BOTTLENECK*
[ ] Define the best summary statistics (or create a test to find them)
[ ] Find the best training parameters (or create a test to find them)
[ ] Build tests for the pipeline with known model

Sub-tasks are to be managed by the main task lead.
High-tasks are to be assigned to team members.
GitHub issues are to be created for each sub-task.
A document is to be created for each high-level task to keep track of progress.
Logs and metrics are to be implemented whenever possible.
Everything should be done in a reproducible way and have the appriopriate level of verbosity available to the user.
The pipeline construction is to be discussed with the whole team to be able to adapt and be coordinated on the interfaces between every part of the pipeline.fdc

## GIT HUB INSTRUMENT 

- Any time time a document is created, it needs to be added to "Name_of_files.md" with a short description of what is that.
- Every time there is a commit, use this structure:
prefix: Comment    
The prefix should be feat for new functionalities or new files created, fix for bugs correction, while the comment should be just a few lines defining what was done.



