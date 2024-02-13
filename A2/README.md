# ELE6310E Assignment 2

First, read all of the assignment questions in the PDF 


## Install timeloop-Accelergy
```
git clone https://github.com/RedaBensaidDS/ELE6310E.git
source ELE6310E/A2/install_timeloop/install_timeloop.sh
# (if necessary)
export PATH=$PATH:~/.local/bin
# make sure timeloop executables can be found
which -a timeloop-model
# make sure accelergy executable can be found
which -a accelergy
```

# Question 1
You can run Accelergy to get the energy consumption using following command:
```
timeloop-model Q1/arch/*.yaml  Q1/prob/*.yaml  Q1/map/Q1_ws.map.yaml
```
You can extract the energy consumption, memory accesses, and all other stats from the `timeloop-model.stats.txt`. 
