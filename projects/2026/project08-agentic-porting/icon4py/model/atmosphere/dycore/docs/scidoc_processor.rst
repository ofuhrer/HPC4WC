Scidoc processor
================

The scidoc processor is an extension of `sphinx's autodoc
documenter <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_
with some additions tailored for ICON4Py long methods.


Using the scidoc processor in the sphinx documentation
------------------------------------------------------
Analogously to sphinx's autodoc, the process to document a class/method/function
simply requires to apply the ``autoscidoc`` keyword instead of ``autodoc``:

.. code-block:: restructuredtext

    .. autoscidoc:: icon4py.model.atmosphere.dycore.solve_nonhydro.SolveNonhydro.run_predictor_step
       :no-index:

The example above runs the processor over the ``run_predictor_step`` method, the
results of which can be found at :doc:`dycore_numerics_nonhydro`.

Adding scidoc documentation to the code
---------------------------------------
The processor is designed in such a way that the source code needs only a few
comment lines for a fully featured documentation page to be automatically
generated.

As an example from the dycore, the comment block describing the call to the
stencil ``_compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates``
results in the web page section displayed in the frame below (and part of the
:doc:`dycore_numerics_nonhydro` page).

All that is required are the **Outputs** and **Inputs** sections, within which 
variables processed by the stencil are listed.
Computation parameters should be described in a few lines where parameters
labels have to match namespace names of the calling method.

The processor takes care of:
 - generating a title that is also a hyperlink to the stencil documentation,
 - parsing the equations with LaTeX syntax and macros,
 - inferring and adding variable types to both inputs and outputs,
 - generating and including figures describing the offset providers used in the stencil,
 - including the source code of the stencil call

.. code-block:: Python

   # scidoc:
   # Outputs:
   #  - z_gradh_exner :
   #     $$
   #     \exnerprimegradh{\ntilde}{\e}{\k} = \Cgrad \Gradn_{\offProv{e2c}} \exnerprime{\ntilde}{\c}{\k}, \quad \k \in [0, \nflatlev)
   #     $$
   #     Compute the horizontal gradient (at constant height) of the
   #     temporal extrapolation of perturbed exner function on flat levels,
   #     unaffected by the terrain following deformation.
   #
   # Inputs:
   #  - $\exnerprime{\ntilde}{\c}{\k}$ : z_exner_ex_pr
   #  - $\Cgrad$ : inverse_dual_edge_lengths
   #
   self._compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
       ...
   )


.. raw:: html

    <div style="border: 2px solid #000000; overflow: hidden; margin: 5px auto; max-width: 1000px;">
    <iframe src="dycore_numerics_nonhydro.html#self-compute-horizontal-gradient-of-exner-pressure-for-flat-coordinates"
            style="border: 0px none; margin-left: -36px; margin-top: 0px; height: 440px; width: 768px;"> <!--768px is the max width without sidebar-->
    </iframe>
    </div>
