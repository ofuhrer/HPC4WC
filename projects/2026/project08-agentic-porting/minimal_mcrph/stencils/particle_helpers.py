"""Field-level versions of the particle helper functions in particles.py.

particles.py's particle_meanmass/particle_diameter/particle_velocity operate on
plain Python floats (used for one-time coefficient setup in Phase 1, no
vertical-field dependence). The process stencils need the same formulas
applied per-level across a column -- that's what these are.
"""

import gt4py.next as gtx
from gt4py.next import maximum, minimum

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import type_alias as ta


@gtx.field_operator
def particle_meanmass(
    q: fa.CellKField[ta.wpfloat],
    n: fa.CellKField[ta.wpfloat],
    x_min: ta.wpfloat,
    x_max: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """Eq. (94) of SB2006, with limiters. mo_2mom_mcrph_processes.f90:365-374."""
    eps = 1.0e-20
    return minimum(maximum(q / (n + eps), x_min), x_max)
