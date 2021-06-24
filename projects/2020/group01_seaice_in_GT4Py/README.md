# GFS Three-layer Thermodynomics Sea-Ice Scheme Module

## Description

Sea ice is a thin skin of frozen water covering the polar oceans. The sea ice strongly interacts with both the
atmosphere above and the ocean underneath in the high latitudes. In a coupled weather/climate system, changes in
sea ice extent, thickness and concentration regionally or globally would influence oceanic and atmospheric
conditions, which in turn affect the sea ice distribution. The physical and dynamical processes affecting the
weather and climate are considered as follows:

The high albedo of the sea ice reflects more solar radiation back to the space. The feedbacks are considered as
positive. The broader the sea ice cover, the higher the surface albedo, which result in less amount of solar
radiation absorbed at the Earth's surface. A cooler surface would favor more sea ice to form. The process would
be reversed in less sea ice situation.
The sea ice restricts the heat/water exchange between the air and ocean. The presence of extensive areas of sea
ice would suppress the heat loss in winter and the heat gain in summer by the ocean. Even a thin ice cover
influences the turbulent heat transfer significantly between ocean and atmosphere. The surface fluxes of sensible
and latent heat can be greater by up to two orders of magnitude at the open water surface of a lead or polynya
than that through (snow covered) pack ice.
The sea ice modifies air/sea momentum transfer, ocean fresh water balance and ocean circulation. The freezing
and melting of the ocean surface and the associated fluxes of salt and heat produce major changes in the density
structure of the polar water. Formation of sea ice injects salt into the ocean makes the water heavier and more
convectively unstable, conversely when melting occurs, stable and fresh layers can prevent deep covective activity.
A sea ice model, in general, may contain subcomponents treating 1) dynamics (ice motion), 2) ice transport,
3) multiple ice thickness categories (including leads), 4) surface albedo, and 5) vertical thermodynamics.
GFS sea ice scheme is concerned with a scheme for the last of these processes. A three-layer thermodynamic sea
ice model (Winton (2000) [160]) has been coupled to GFS. It predicts sea ice/snow thickness, the surface temperature
and ice temperature structure. In each model grid box, the heat and moisture fluxes and albedo are treated
separately for the ice and the open water.

[160] M. Winton. A reformulated three-layer sea ice model. J. Atmos. Oceanic Tech., 17:525–531, 2000.

## GFS Sea Ice Driver General Algorithm

The model has four prognostic variables: the snow layer thickness hs, the ice layer thickness hi, the upper
and lower ice layer temperatures located at the midpoints of the layers hi/4 and 3hi/4 below the ice surface,
respectively T1 and T2. The temperature of the bottom of the ice is fixed at Tf, the freezing temperature of
seawater. The temperature of the top of the ice or snow, Ts, is determined from the surface energy balance.
The model consists of a zero-heat-capacity snow layer overlying two equally thick sea ice layers (Figure 1).
The upper ice layer has a variable heat capacity to represent brine pockets.

Fig.1 Schematic representation of the three-layer model
The ice model main program ice3lay() performs two functions:

Calculation of ice temperature 
The surface temperature is determined from the diagnostic balance between the upward conduction of heat through
snow and/or ice and upward flux of heat from the surface.
Calculation of ice and snow changes 
In addition to calculating ice temperature changes, the ice model must also readjust the sizes of the snow and
ice layers 1) to accommodate mass fluxes at the upper and lower surfaces, 2) to convert snow below the water
line to ice, and 3) to equalize the thickness of the two ice layers.

## Three-layer Thermodynamics Sea Ice Model General Algorithm

- Ice temperature calculation.
- Calculate the effective conductive coupling of the snow-ice layer between the surface and the upper layer ice temperature hi/4 beneath the snow-ice interface (see eq.(5) in Winton (2000) [160]).
- Calculate the conductive coupling between the two ice temperature points (see eq.(10) in Winton (2000) [160]).
- Calculate the new upper ice temperature following eq.(21) in Winton (2000) [160].
- If the surface temperature is greater than the freezing temperature of snow (when there is snow over) or sea ice (when there is none), the surface temperature is fixed at the melting temperature of snow or sea ice, respectively, and the upper ice temperature is recomputed from eq.(21) using the coefficients given by eqs. (19),(20), and (18). An energy flux eq.(22) is applied toward surface melting thereby balancing the surface energy budget.
- Calculate the new lower ice temperature following eq.(15) in Winton (2000) [160].
- Calculate the energy for bottom melting (or freezing, if negative) following eq.(23), which serves to balance the difference between the oceanic heat flux to the ice bottom and the conductive flux of heat upward from the bottom.
- Calculation of ice and snow mass changes.
- Calculate the top layer thickness.
- When the energy for bottem melting Mb is negative (i.e., freezing is happening),calculate the bottom layer thickness h2 and the new lower layer temperature (see eqs.(24)-(26)).
- If ice remains, even up 2 layers, else, pass negative energy back in snow. Calculate the new upper layer temperature (see eq.(38)).
