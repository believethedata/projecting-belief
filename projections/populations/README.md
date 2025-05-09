# Population projection model
The population projection models use ONS projections for population by age, sex and country. They apportion these projected country populations to Local District Authorities (LDAs). This process involves two steps:
 1. We build a basic population projection using historical data and observed "survival" rates (which actually capture migration as well as mortality). New births are projected assuming a constant birth rate.
 2. The projections of this model are scaled linearly to ensure that country totals sum to the ONS projections for each sex and age band.
