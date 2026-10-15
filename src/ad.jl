"""
    ADModelBackend(gradient_backend, hprod_backend, jprod_backend, jtprod_backend, jacobian_backend, hessian_backend, ghjvprod_backend, hprod_residual_backend, jprod_residual_backend, jtprod_residual_backend, jacobian_residual_backend, hessian_residual_backend)

Structure that define the different backend used to compute automatic differentiation of an `ADNLPModel`/`ADNLSModel` model.
The different backend are all subtype of `ADBackend` and are respectively used for:
  - gradient computation;
  - hessian-vector products;
  - jacobian-vector products;
  - transpose jacobian-vector products;
  - jacobian computation;
  - hessian computation;
  - directional second derivative computation, i.e. gᵀ ∇²cᵢ(x) v.

The default constructors are 
    ADModelBackend(nvar, f, ncon = 0, c = (args...) -> []; show_time::Bool = false, kwargs...)
    ADModelNLSBackend(nvar, F!, nequ, ncon = 0, c = (args...) -> []; show_time::Bool = false, kwargs...)

If `show_time` is set to `true`, it prints the time used to generate each backend.

The remaining `kwargs` are either the different backends as listed below or arguments passed to the backend's constructors:
  - `gradient_backend = ForwardDiffADGradient`;
  - `hprod_backend = ForwardDiffADHvprod`;
  - `jprod_backend = ForwardDiffADJprod`;
  - `jtprod_backend = ForwardDiffADJtprod`;
  - `jacobian_backend = SparseADJacobian`;
  - `hessian_backend = ForwardDiffADHessian`;
  - `ghjvprod_backend = ForwardDiffADGHjvprod`;
  - `hprod_residual_backend = ForwardDiffADHvprod` for `ADNLSModel` and `EmptyADbackend` otherwise;
  - `jprod_residual_backend = ForwardDiffADJprod` for `ADNLSModel` and `EmptyADbackend` otherwise;
  - `jtprod_residual_backend = ForwardDiffADJtprod` for `ADNLSModel` and `EmptyADbackend` otherwise;
  - `jacobian_residual_backend = SparseADJacobian` for `ADNLSModel` and `EmptyADbackend` otherwise;
  - `hessian_residual_backend = ForwardDiffADHessian` for `ADNLSModel` and `EmptyADbackend` otherwise.

"""
struct ADModelBackend{GB, HvB, JvB, JtvB, JB, HB, GHJ, HvBLS, JvBLS, JtvBLS, JBLS, HBLS}
  gradient_backend::GB
  hprod_backend::HvB
  jprod_backend::JvB
  jtprod_backend::JtvB
  jacobian_backend::JB
  hessian_backend::HB
  ghjvprod_backend::GHJ

  hprod_residual_backend::HvBLS
  jprod_residual_backend::JvBLS
  jtprod_residual_backend::JtvBLS
  jacobian_residual_backend::JBLS
  hessian_residual_backend::HBLS
end

function ADModelBackend(;
  forward_backend::ADTypes.AbstractADType = ADTypes.AutoForwardDiff(), 
  reverse_backend::ADTypes.AbstractADType = ADTypes.AutoReverseDiff(),
  second_order_backend::ADTypes.AbstractADType = SecondOrder(forward_backend, reverse_backend)
)
  check_backends(forward_backend, reverse_backend, second_order_backend)
  return ADModelBackend(
    reverse_backend,      # gradient         
    second_order_backend, # hprod            
    forward_backend,      # jprod            
    reverse_backend,      # jtprod           
    forward_backend,      # jacobian         
    second_order_backend, # hessian          
    forward_backend,      # ghjvprod         
    EmptyADbackend(),     # hprod_residual   
    EmptyADbackend(),     # jprod_residual   
    EmptyADbackend(),     # jtprod_residual  
    EmptyADbackend(),     # jacobian_residual
    EmptyADbackend(),     # hessian_residual 
  )
end

"""
    ADModelNLSBackend(; forward_backend, reverse_backend, second_order_backend)

Same as [`ADModelBackend`](@ref) for an `ADNLSModel`. The derivatives of the objective
`½‖F(x)‖²` are not computed by AD but through the residual `F`, so the `gradient`, `hprod`,
`hessian` and `ghjvprod` backends are empty. The residual backends are:
  - `jprod_residual_backend = forward_backend`;
  - `jtprod_residual_backend = reverse_backend`;
  - `hprod_residual_backend = second_order_backend`;
  - `jacobian_residual_backend = forward_backend`;
  - `hessian_residual_backend = second_order_backend`.

The `jprod`, `jtprod` and `jacobian` backends are the ones of the (optional) constraints `c`.
"""
function ADModelNLSBackend(;
  forward_backend::ADTypes.AbstractADType = ADTypes.AutoForwardDiff(),
  reverse_backend::ADTypes.AbstractADType = ADTypes.AutoReverseDiff(),
  second_order_backend::ADTypes.AbstractADType = SecondOrder(forward_backend, reverse_backend),
)
  check_backends(forward_backend, reverse_backend, second_order_backend)
  return ADModelBackend(
    EmptyADbackend(),     # gradient
    EmptyADbackend(),     # hprod
    forward_backend,      # jprod
    reverse_backend,      # jtprod
    forward_backend,      # jacobian
    EmptyADbackend(),     # hessian
    EmptyADbackend(),     # ghjvprod
    second_order_backend, # hprod_residual
    forward_backend,      # jprod_residual
    reverse_backend,      # jtprod_residual
    forward_backend,      # jacobian_residual
    second_order_backend, # hessian_residual
  )
end

function check_backends(
  forward_backend::ADTypes.AbstractADType,
  reverse_backend::ADTypes.AbstractADType,
  second_order_backend::ADTypes.AbstractADType
)
  @assert ADTypes.mode(forward_backend) isa
      Union{ADTypes.ForwardMode,ADTypes.ForwardOrReverseMode}
  @assert ADTypes.mode(reverse_backend) isa
      Union{ADTypes.ReverseMode,ADTypes.ForwardOrReverseMode}
  @assert second_order_backend isa SecondOrder
end