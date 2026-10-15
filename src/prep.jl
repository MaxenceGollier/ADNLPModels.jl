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

  ob = one(eltype(x))
  lhprod_prep = DI.prepare_hvp(ℓ, adbackend.hprod_backend, x, (v,), Cache(cx), Constant(y), Constant(ob))
  hprod_prep = DI.prepare_hvp(f, adbackend.hprod_backend, x, (v,))

  jprod_prep = DI.prepare_pushforward(c!, cx, adbackend.jprod_backend, x, (v,))
  jtprod_prep = DI.prepare_pullback(c!, cx, adbackend.jtprod_backend, x, (w,))

  # gᵀ ∇²cᵢ(x) v for all i, see `ShiftedConstraint`. `g` and `v` are both of the size of `x`.
  ghjvprod = if length(cx) > 0
    DI.prepare_second_derivative(
      φ, adbackend.ghjvprod_backend, zero(eltype(x)), Constant(x), Constant(v), Constant(v),
    )
  else
    nothing
  end

  return ADModelPrep(
    gradient_prep,
    hprod_prep,
    lhprod_prep,
    jprod_prep,
    jtprod_prep,
    ghjvprod,

    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
  )
end