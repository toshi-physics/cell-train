Equations being implemented in this are
dt(e_p) = l^2 dx^2 (e_p+e_a)
tau_2 dt(e_a) = - e_a + alpha c
tau_3 dt c = -c + beta * tanh(e_p)

There are three external load cases. One where we study steady states the other where we study the response to step strain. A third where we study response to oscillatory strain.


Under each external loading, we first fix system size and then look at parameter regimes where alpha beta '<' critical and '>' than  critical value.

First generate steady state oscillations files by running model_B.py -p steady_state. Change parameters in parameters.json file.

For the penetration length graph you need to use data which applied force of 1.0 delta and use file penetration_length_dump.ipynb

For the resonance graph you need to use data where perturbation is proportional to length of the system. Osc_strain_prop_disp is that data.

Analysis of square penetration length is using the appropriate files 

Harmonics plot is made using analysis_harmonics.py


