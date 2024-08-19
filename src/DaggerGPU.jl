module DaggerGPU

using Dagger, MemPool
using Distributed
using KernelAbstractions, Adapt

import Dagger: Chunk
import LinearAlgebra

const CPUProc = Union{OSProc, Dagger.ThreadProc}

struct Kernel{F} end
Kernel(F) = Kernel{F}()

function (::Kernel{F})(args...; ndrange) where F
    @nospecialize args
    dev = kernel_backend()
    kern = F(dev)
    kern(args...; ndrange)
    KernelAbstractions.synchronize(dev)
end

macro gpuproc(PROC, T)
    PROC = esc(PROC)
    T = esc(T)
    quote
        # Assume that we can run anything
        Dagger.iscompatible_func(proc::$PROC, opts, f) = true
        Dagger.iscompatible_arg(proc::$PROC, opts, x) = true

        # CPUs shouldn't process our array type
        Dagger.iscompatible_arg(proc::Dagger.ThreadProc, opts, x::$T) = false
    end
end

processor(kind::Symbol) = processor(Val(kind))
processor(::Val) = Dagger.ThreadProc
cancompute(kind::Symbol) = cancompute(Val(kind))
cancompute(::Val) = false
function with_device end

move_optimized(from_proc::Dagger.Processor,
               to_proc::Dagger.Processor,
               x) = nothing

kernel_backend() = kernel_backend(Dagger.thunk_processor())
kernel_backend(::Dagger.ThreadProc) = CPU()

using Requires
@static if !isdefined(Base, :get_extension)
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include(joinpath(@__DIR__, "..", "ext", "CUDAExt.jl"))
    end
    @require AMDGPU="21141c5a-9bdb-4563-92ae-f87d6854732e" begin
        include(joinpath(@__DIR__, "..", "ext", "ROCExt.jl"))
    end
    @require Metal="dde4c033-4e86-420c-a63e-0dd931031962" begin
        include(joinpath(@__DIR__, "..", "ext", "MetalExt.jl"))
    end
end
end

end
