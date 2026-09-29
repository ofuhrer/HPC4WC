---
tags:
  - physics components
  - tendencies
  - model state
---

# Physics components interface: return values

- **Status**: valid
- **Authors**: Magdalena Luz (@halungge), Ong Chia Rui (@ongchia)
- **Created**: 2024-07-23
- **Updated**: 2024-09-11

In the context of a modular python driven model, facing the question of standardizing components interfaces
we decided for physics modules always returning _tendencies_ and
not allowing direct update of model state (prognostic, diagnostics), to achieve more transparency as to where state is updated and allowing the user to experiment
with the coupling of components.

## Context

We aim at making `icon4py` a flexible model code which endorses modularity and extensibility, and makes it
easy for future users and developers to add new components and replace existing ones. This includes the
possibility of experimenting with the coupling of different physics components, revisiting
the order in which they are run and whether or not their result should be applied to the model state.

In the original [ICON code](https://gitlab.dkrz.de/icon/icon-model) physics components may do both: compute tendencies as well as
update the model state (update diagnostic and prognostic variables directly). More precisely, physics
processes are split in fast physics (runs every timestep) and slow physics (runs every n-th timestep, n > 1). Fast physics update model state directly
while slow physics return tendendencies. The distinction is somewhat arbitrary and probably largely dictated by
computational efficiency of a component.

We believe that this distinction is not really transparent, makes the code inflexible and the individual components and their application order tightly coupled.

We want to provide a cleaner and more transparent structure that treats all physics components in the same way and allows
the user to decide which update should be applied to the model state prior to the next dynamics step.

**Concerns**

- If a component depends on the tendencies produced by another component, it is unclear whether the effect of this other component
  can be cleanly isolated.

## Decision

All **physics components return tendencies**. It is **forbidden to ever change input diagnostic and prognostic
fields** inside a component. The model state is updated in a separate step after all tendencies have been computed. Where necessary
tendencies from a component `f` might be passed to another component `h` as input.
In that step, the developer can decide which tendencies (produced by which component) need to be applied to the state.

Let's consider three physics components `f`, `g`, `h` which compute tendencies for the fields `rho`, `theta`, `T`.

```python
state = (rho, theta, w, T) # contains diagnostic and prognostic variables at time t

f(rho, T, w) -> ddt_rho, ddt_T # rho, T, w are input only
g(rho, T, theta) -> ddt_theta, ddt_T # rho, theta, are input only
h(rho, T, ddt_T) -> ddt_T # rho, theta, ddt_theta are input only

```

All three components produce tendencies for `T`, these need to be kept separate until the state is finally updated.
Components might depend one each other which can be made transparent through a dependency on the tendencies: in the example
`h` might incorporate changes from `f` and `g` through the dependencies on `ddt_T`. This dependency has to be passed to the component `h` as
explicit argument. The component `h` can then decide how to incorporate the tendencies from `f` and `g`.
Hence `ddt_T` needs to be a struct that lets the user access the tendencies computed by `f` and `g` individually.

## Consequences

On the positive side:

- It is easier to understand what exactly is in the model state and what tendencies issued from which
  physics component are used where.

On the negative side:

- Several physics components might produce tendencies for the same field (as in the example above): for example microphysics and radiation
  produce tendencies for temperature. These individual tendencies need to be kept apart until the state is finally updated
  prior to the next dynamics step. This leads to a more complex structure of the model state, especially the tendency fields.
- Keeping tendencies produced by the individual components separate is more memory intensive
  and requires an explicit update step.

## Alternatives considered

### updating state (prognostics, diagnostics) directly inside a component

```python
state = (rho, theta, w, T) # contains diagnostic and prognostic variables at time t

f(rho, T, w) -> rho, T # rho, T, w are input only
g(rho, T, theta) -> theta, T # rho, theta, are input only
h(rho, T) -> T # rho, theta, ddt_theta are input only
```

- Good, because no extra handling and bookkeeping of tendencies is needed.
- Bad, because it is unclear outside of a component whether state has changed, and what changes it incorporates.
- Bad, because the input value for a component changes on depending on the order of the components.

### mixture of both approaches

```python
state = (rho, theta, w, T) # contains diagnostic and prognostic variables at time t

f(rho, T, w) -> ddt_rho, ddt_T # rho, T, w are input only
g(rho, T, theta) -> ddt_theta, T # rho, theta, are input only
h(rho, T) -> T # rho, theta, ddt_theta are input only
```

- Bad, unclear interface.
