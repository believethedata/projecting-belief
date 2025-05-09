# Model selection
The notebook in this folder uses data from the 2001 and 2011 census surveys in England & Wales to project religious mix trends forward to 2021. (Data from Scotland and Northern Ireland was not available in a suitable format for 2001.)

The basic units of analysis were cohorts of people born in 5-year bands, by geographic area (LDA) and sex.

## The over-tens model
For cohorts born before 2001, we compared their religious mix in 2001 with that in 2011*. We then extrapolated this trend forward to 2021, using a variety of approaches:
 - Linear extrapolation
 - Exponential extrapolation
 - Transition-matrix-based "process extrapolation"

We also applied the national trend to 2011 data to project forward to 2021, leading to a further three projections.

The accuracy of the different extrapolation approaches used was assessed by comparing these projections to the actual religious mix observed in the 2021 census. We then compose an ensemble model combining these individual extrapolations, with weights chosen to minimise error on a 50% training set of LDAs.
 
*NB: We recognise that these populations will not be identical at both points in time. Changes in religious mix will be affected by migration and death as well as changing affiliations of the population that is retained. The model implicitly assumes that the drivers of change will remain stable.


## The under-tens model
Children who were under ten in 2011 hadn't been born in 2001. To predict the likely religious mix of newborns, we recognise that their parents or carers will fill the census in for them. Making use of the same set of models that we used to extrapolate temporal trends within individual birth cohorts, in this case we model the difference between age groups, as a proxy for modelling the difference between parental affiliations and those of the children on whose behalf they are responding. We do this in two different ways:
 - Use 2001 "parental" religious mix data to predict the religious mix of 2011 newborns
 - Use 2011 projections of "parental" mix from the over-ten model to predict the religious mix of 2011 newborns

To keep the model (slightly) simpler, we pick the best performing cohort-model pair as opposed to creating an ensemble.

## Outputs
The two outputs of this stage are:
 - a set of weights determining the optimal combination of the projections for the over-tens model
 - a model-cohort pair for each set of female/male 0-4/5-9 year old newborns, chosen to minimise the error