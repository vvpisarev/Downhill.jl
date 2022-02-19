# Optimization Methods

Optimization methods are defined as subtypes `AbstractOptBuffer` type. 
The structs holding the data required for core iteration logic should subtype 
 `OptBuffer <: AbstractOptBuffer`. 
Stopping criteria, logging, step limitations etc. are implemented as subtypes of 
`Wrapper <: AbstractOptBuffer`.

## Core Optimization Methods
```@docs
FixedRateDescent

MomentumDescent

NesterovMomentum

SteepestDescent

HyperGradDescent

CGDescent

BFGS

CholBFGS
```
