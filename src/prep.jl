struct ADModelPrep{GP, HvP, JvP, JtvP, JP, HP, GHJP, HvPLS, JvPLS, JtvPLS, JPLS, HPLS}
  gradient_prep::GP
  hprod_prep::HvP
  jprod_prep::JvP
  jtprod_prep::JtvP
  jacobian_prep::JP
  hessian_prep::HP
  ghjvprod_prep::GHJP

  hprod_residual_prep::HvPLS
  jprod_residual_prep::JvPLS
  jtprod_residual_prep::JtvPLS
  jacobian_residual_prep::JPLS
  hessian_residual_prep::HPLS
end

function ADModelPrep(adbackend::ADModelBackend, f::Function, c!::Function, ℓ::LagrangianFunction, x::AbstractVector)

  # Construct Lagrangian function
  gradient_prep = DifferentiationInterface.prepare_gradient(f, adbackend.gradient_backend, x)
  hprod_prep = DifferentiationInterface.prepare_hvp(ℓ, adbackend.hprod_backend, x, Cache(cx), Constant(y), Constant(ob))
  return ADModelPrep(
    gradient_prep,
    hprod_prep,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,

    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
  )
end