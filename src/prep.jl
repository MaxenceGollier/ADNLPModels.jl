struct ADModelPrep{GP, HvP, LHvP, JvP, JtvP, GHJP, HvPLS, JvPLS, JtvPLS, JPLS, HPLS}
  gradient_prep::GP
  hprod_prep::HvP
  lhprod_prep::LHvP
  jprod_prep::JvP
  jtprod_prep::JtvP
  ghjvprod_prep::GHJP

  hprod_residual_prep::HvPLS
  jprod_residual_prep::JvPLS
  jtprod_residual_prep::JtvPLS
  jacobian_residual_prep::JPLS
  hessian_residual_prep::HPLS
end

function ADModelPrep(
  adbackend::ADModelBackend, 
  f::OF, 
  c!::CF, 
  ℓ::LagrangianFunction{OF, CF}, 
  φ::ShiftedConstraint{CF},
  x::AbstractVector,
  cx::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  w::AbstractVector,
) where{OF, CF}

  gradient_prep = DI.prepare_gradient(f, adbackend.gradient_backend, x)
  hprod_prep = DI.prepare_hvp(f, adbackend.hprod_backend, x, (v,))

  lhprod_prep, ghjvprod_prep, jprod_prep, jtprod_prep = if length(cx) > 0
    (
      DI.prepare_hvp(ℓ, adbackend.hprod_backend, x, (v,), Cache(cx), Constant(y), Constant(one(eltype(x)))),
      DI.prepare_second_derivative(φ, adbackend.ghjvprod_backend, zero(eltype(x)), Constant(x), Constant(v), Constant(v)),
      DI.prepare_pushforward(c!, cx, adbackend.jprod_backend, x, (v,)),
      DI.prepare_pullback(c!, cx, adbackend.jtprod_backend, x, (w,)),
    )
  else
    (nothing, nothing, nothing, nothing)
  end

  return ADModelPrep(
    gradient_prep,
    hprod_prep,
    lhprod_prep,
    jprod_prep,
    jtprod_prep,
    ghjvprod_prep,

    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
  )
end

"""
    ADModelPrep(adbackend, F!, c!, ψ, x, Fx, cx, v, w, u)

Preparation for an `ADNLSModel`, where `F!` is the residual (`nequ` outputs) and `c!` the
nonlinear constraints (`nnln` outputs). `ψ = DotConstraints(F!)`, i.e. `ψ(x, Fx, u) = uᵀF(x)`.

The objective `½‖F(x)‖²` has no preparation: its derivatives are computed through the residual.
The remaining arguments are only used for their type and size: `Fx`, `u` (`nequ`), `cx`, `w`
(`nnln`) and `x`, `v` (`nvar`).
"""
function ADModelPrep(
  adbackend::ADModelBackend,
  F!::OF,
  c!::CF,
  ψ::DotConstraints{OF},
  x::AbstractVector,
  Fx::AbstractVector,
  cx::AbstractVector,
  v::AbstractVector,
  w::AbstractVector,
  u::AbstractVector,
) where {OF, CF}

  # Residual F!: x ↦ F(x)
  jprod_residual_prep =
    DI.prepare_pushforward(F!, Fx, adbackend.jprod_residual_backend, x, (v,))
  jtprod_residual_prep =
    DI.prepare_pullback(F!, Fx, adbackend.jtprod_residual_backend, x, (u,))
  
    # Σᵢ uᵢ ∇²Fᵢ(x) v. With u = eᵢ, this is ∇²Fᵢ(x) v.
  hprod_residual_prep = DI.prepare_hvp(
    ψ, adbackend.hprod_residual_backend, x, (v,), Cache(Fx), Constant(u),
  )

  jprod_prep, jtprod_prep = if length(cx) > 0
    (
      DI.prepare_pushforward(c!, cx, adbackend.jprod_backend, x, (v,)),
      DI.prepare_pullback(c!, cx, adbackend.jtprod_backend, x, (w,)),
    )
  else
    (nothing, nothing)
  end

  return ADModelPrep(
    nothing,
    nothing,
    nothing,
    jprod_prep,
    jtprod_prep,
    nothing,

    hprod_residual_prep,
    jprod_residual_prep,
    jtprod_residual_prep,
    nothing,
    nothing,
  )
end
