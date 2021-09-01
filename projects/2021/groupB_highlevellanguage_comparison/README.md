# HPC4WC
Project repo of Group B for the lecture HPC4WC (High Performance Computing for Weather and Climate) course at ETH.


Various scientific fields have benefited greatly from easy-to-use domain specific languages (DSL). GT4Py aims offer a similar high-performance, high-level solution for weather and climate simulations. It implements an extension of Python which generates efficient implementations of finite difference stencil computations for a selection of backends and architectures. In this report we will explore the performance of GT4Py and how it compares to a Machine Learning DSL and several general purpose optimized NumPy implementations. We find that GT4Py is superior to the other options in a variety of situations and offers a 4x speedup on CPUs and 2x speedup on GPUs when compared to the next best library that we test.

© Safira Piasko, Tom Lausberg, Tobia Claglüna
