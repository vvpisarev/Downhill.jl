# Customization

```@meta
CurrentModule = Downhill
```

Subtypes of `Downhill.Wrapper` are used to add custom behavior to optimization, 
such as stopping after reaching certain criteria, limiting timesteps etc.

A wrapper around an object of `AbstractOptBuffer` type should at a minimum provide 
the `base_method(M::Wrapper)` to return the wrapped object.

The wrappers can modify the optimization behavior by overloading the following methods:
```@docs
init!

callfn!

reset!

step!

stopcond

conv_success
```

## Predefined Wrappers

```@docs
BasicConvergenceStats

ConstrainStepSize
```