"""
Shared helpers for the `DifferentiationInterface.jl` (DI) backends.

ADNLPModels uses DI as a *dense* AD kernel only: sparsity pattern detection
(`SparseConnectivityTracer`) and coloring/decompression (`SparseMatrixColorings`) are
handled by ADNLPModels itself, because the sparse Jacobians and Hessians are stored in a
single triangle, which DI's `AutoSparse` cannot produce.
"""

# The `AbstractADType`s wrapped by the backends. `NC` variants ("no compile") are used
# wherever the tape cannot be reused: either the element type varies at run time (the
# `Generic*` backends) or the differentiated function changes between calls.
const FDBackend = AutoForwardDiff()
const RDBackend = AutoReverseDiff(compile = Val(true))
const RDBackendNC = AutoReverseDiff()

# Forward-over-reverse: the cheapest second-order composition for Hessian-vector products.
const FoRBackend = SecondOrder(AutoForwardDiff(), AutoReverseDiff(compile = Val(true)))
const FoRBackendNC = SecondOrder(AutoForwardDiff(), AutoReverseDiff())

"""
    LagrangianFunction(f, c!, ncon)

Callable computing `ℓ(x, cx, y, ob) = ob * f(x) + dot(c(x), y)`, with `cx` a scratch buffer
for the in-place constraints `c!`.

`y` and `ob` are passed to DI as `Constant` contexts and `cx` as a `Cache`, so that a single
preparation covers every Hessian call: the Lagrangian (`y`, `ob`), the objective alone
(`y = 0`), and the `j`-th constraint or residual (`y = eⱼ`, `ob = 0`).

This replaces the augmented function over `z = (cx, x, y, ob)` of length `nvar + 2 * ncon + 1`
that ADNLPModels used to build by hand: differentiating with respect to `x` alone is both
simpler and cheaper.

A named struct is used rather than a closure because DI records `typeof(f)` in the
preparation signature, and a closure's type would vary with the type of the captured `y`.
"""
struct LagrangianFunction{F, C}
  f::F
  c!::C
  ncon::Int
end

function (ℓ::LagrangianFunction)(x, cx, y, ob)
  if ℓ.ncon > 0
    ℓ.c!(cx, x)
    return ob * ℓ.f(x) + dot(cx, y)
  else
    return ob * ℓ.f(x)
  end
end

"""
    DotConstraints(c!)

Callable computing `ψ(x, cx, u) = dot(c(x), u)`, whose gradient with respect to `x` is the
transposed Jacobian-vector product `J(x)ᵀu`.

Used instead of `DI.pullback` for forward-mode backends: DI implements the pullback of a
forward-mode backend as one pushforward per component of `x`, which costs `nvar` evaluations.
"""
struct DotConstraints{C}
  c!::C
end

(ψ::DotConstraints)(x, cx, u) = (ψ.c!(cx, x); dot(cx, u))

"""
    seeded_duals(tag, x0, nvar)

Allocate a `Vector{ForwardDiff.Dual{tag, T, 1}}` of length `nvar` whose primal parts are
initialised from `x0` and whose partials are zero.

`ReverseDiff.GradientTape` *executes* the function while recording the tape, so its input
must be a well-defined point. Allocating it with `undef` leaves the primal parts holding
whatever the allocator hands back, which may legally be non-finite (a recycled `-Inf` bounds
array, say). Domain-restricted primitives such as `cos`, `sqrt` or `log` then throw during
tape construction, so the backend fails to build at all. See issue #383.

Recording at a fixed point also makes tape compilation deterministic.
"""
function seeded_duals(::Type{tag}, x0::AbstractVector{T}, nvar::Integer) where {tag, T}
  z = Vector{ForwardDiff.Dual{tag, T, 1}}(undef, nvar)
  @inbounds for i = 1:nvar
    z[i] = ForwardDiff.Dual{tag}(x0[i], zero(T))
  end
  return z
end

# Temporary support for `DI.Cache` contexts with `AutoReverseDiff`, restricted to
# ADNLPModels' own `LagrangianFunction`.
#
# Why this is fragile: it relies on DI internals (`prepare_gradient_nokwarg`, `signature`,
# `check_prep`, `fix_tail`, `GradientPrep`), which may change in any DI release.
# Keep the DI compat bound tight while this file exists.
#
# The plan is to delete this file once the same logic lives in DI's ReverseDiff extension
# (there, the methods would accept any `f` instead of `f::LagrangianFunction`).
#

# Preparation result: ReverseDiff's config, plus one prepared buffer per `Cache` context
# (`nothing` for the other contexts).
struct RDCacheGradientPrep{SIG, CFG, CT} <: DI.GradientPrep{SIG}
  _sig::Val{SIG}
  config::CFG
  contexts_tracked::CT
end

# Element type of the numbers ReverseDiff produces when running `f` on `cfg.input`.
# `Nothing` means "not read directly from x", which is what arithmetic results are.
function tracked_eltype(cfg::ReverseDiff.GradientConfig)
  V = ReverseDiff.valtype(cfg.input)
  D = ReverseDiff.derivtype(cfg.input)
  return ReverseDiff.TrackedReal{V, D, Nothing}
end

# At preparation: allocate a buffer of tracked numbers for each Cache.
# Filled with zeros so that reading an entry before writing it does not throw.
_to_tracked(::Type, ::DI.Context) = nothing
_to_tracked(T::Type, c::DI.Cache) = fill!(similar(DI.unwrap(c), T), zero(T))

# At each call: use the prepared buffer for a Cache, the plain value otherwise.
_from_prep(c::DI.Context, _) = DI.unwrap(c)
_from_prep(::DI.Cache, buffer) = buffer

function DI.prepare_gradient_nokwarg(
  strict::Val,
  f::LagrangianFunction,
  backend::ADTypes.AutoReverseDiff,
  x,
  contexts::Vararg{DI.Context, C},
) where {C}
  _sig = DI.signature(f, backend, x, contexts...; strict)
  config = ReverseDiff.GradientConfig(x)
  T = tracked_eltype(config)
  contexts_tracked = map(c -> _to_tracked(T, c), contexts)
  return RDCacheGradientPrep(_sig, config, contexts_tracked)
end

function DI.gradient(
  f::LagrangianFunction,
  prep::RDCacheGradientPrep,
  backend::ADTypes.AutoReverseDiff,
  x,
  contexts::Vararg{DI.Context, C},
) where {C}
  DI.check_prep(f, prep, backend, x, contexts...)
  fc = DI.fix_tail(f, map(_from_prep, contexts, prep.contexts_tracked)...)
  return ReverseDiff.gradient(fc, x, prep.config)
end

function DI.gradient!(
  f::LagrangianFunction,
  grad,
  prep::RDCacheGradientPrep,
  backend::ADTypes.AutoReverseDiff,
  x,
  contexts::Vararg{DI.Context, C},
) where {C}
  DI.check_prep(f, prep, backend, x, contexts...)
  fc = DI.fix_tail(f, map(_from_prep, contexts, prep.contexts_tracked)...)
  return ReverseDiff.gradient!(grad, fc, x, prep.config)
end