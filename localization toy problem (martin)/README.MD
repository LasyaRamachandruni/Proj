
# Link Failure Localization in Hierarchical, Multidomain Optical Networks
This is Martin Mihailov Bojinov's project.

It focusses on the architectures needed to simulate monitoring in a multidomain, hierarchical, optical network. It proposes a system of monitoring trails to monitor network traffic in an abstracted broker graph. Three heuristics are proposed to select paths in this system. The goal is to be able to localize failing links based on a selection of monitoring paths. It also simulates OSNR data in GNPy to analyze when soft failure may be occurring. An LSTM is used to predict when individual monitoring trails may be failing, enhancing the architectures' detection capabilities.

## Structure

The project structure is as follows:
- version 1
  > This is the early prototype of the project done in CS297. It can be largely ignored however it is saved here for future reference.
- version 2
  > This is the working, final project, including all its systems.
  - graphics
    > This folder includes images generated to display the topologies
  - hard failure
    - gnpy path selection data
      > This folder contains the data used in our GNPy path selection experiments (input and output).

      > input: GNPy output files simulate optical traffic across multiple time steps. Each corresponding GNPy output file is read in the topX.py files (where X is the topology number).

      > output: The experiment results are a folder of excel files, detailing simulated heuristic's LNI scores. These are plotted in the plots folder.
    - random endpoint selection data
      > This folder contains the data used in preliminary random endpoint experiments. The data here is not used in our final project but serves as a historical experiment.

      > input: there are no input files. In the experiments, endpoints are selected at random and the heuristics are used to select points between the endpoints. You can see early version of this in version1.

      > output: The experiment results are a folder of csv/excel files, detailing simulated heuristic's LNI scores. These are plotted in the plots folder.
  - soft failure
    - gnpy soft failure data
      > This is our GNPy simulation of soft failure data. The is produced using main2.py and service_generator2.py from the fiber_failure branch. They are located in Soham's project files. The data is organized based on "{loss cofficient}-{timesteps ran}".
    - lstm results
      > These are the results of training the LSTM on our GNPy data. The final results were ran on "0.2,0.5-200".
  - topology files
    > This is a folder of all of the .gml topology files used in our experiments.
  - boxplot.py
    > This script was used to plot results using matplotlib.
  - optical_monitoring.py
    > This is our main library that I developed for use in this project. There are 3 main data structures: edge stats, monitoring_trails, and optical monitoring. optical monitoring utilizes the prior data structures and has a list of functions to simulate and localize failure. The actual experiments are done in the top1.py, top2.py, and top3.py files respectively. They call this library.
  - soft_failure_detection.py
    > This is our experiment for soft failure detection. The data from the soft failure folder is read in here and the LSTM is trained and tested at predicting the occurrence of soft failure in a GNPy simulation.
  - top1.py
  - top2.py
  - top3.py
