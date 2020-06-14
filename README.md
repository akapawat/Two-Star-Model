# Two-Star-Model
This is a program to sample Gibbs distribution of the two-star model, one of the model in Exponential Random Graph Model (ERGM) family, using metropolis algorithm. This is part of my undergraduate thesis at Chulalongkorn University. The aim of this program is to reproduce the study of the two-star in dense regime and sparse regime, done by Park and New man 2004 and Annibale and Courtney 2015 respectively. However, it can run other model in ERGM family with only small adjudments. If you're interested in my work or have any question you can contact me directly.
There are three Python file in total. dense_regime.py is coded to run a simulation in dense regime of the two-star model and sparse_regime.py is coded to run a simulation in sparse regime. Although these two use the same set of functions, I write it separately for convenient. After running the simulation, the results will be saved in the same directory as the program, the adjacency matrix will also be saved in separated directory saved matrix. dense_regime.py has a build-in display function plot_data() so you can display the result with the same program. However, to display a result from sparse_regime.py you need to lunch print_k_k2.py.
