# About

Protoyping a router for [JuPedSim](jupedsim.org). 

Plan is to use the metrics produced by [depthmapX](https://github.com/spatialnous/depthmapX) and especially the the library 
[salalib](https://github.com/spatialnous/salalib)

In this repository we have the following files:


- VGA_space_syntax.py to generate metrics and save to csv
- enhanced_simulation_with_grid_routing.py:
   - read a csv file with space-syntax metrics
   - perform a simulation using an isovist-based router. 
- integration_space_syntax_integration: This is just testing different integration definitions
- streamlit app for visualising the csv file and performing some basic routing and testing.
