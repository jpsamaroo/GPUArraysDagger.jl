module DaggerGPU

using Dagger, MemPool, Requires, Adapt
using Distributed
using KernelAbstractions

import Dagger: Chunk

macro gpuproc(PROC, T)
    quote
        # Assume that we can run anything
        Dagger.iscompatible_func(proc::$PROC, opts, f) = true
        Dagger.iscompatible_arg(proc::$PROC, opts, x) = true

        # CPUs shouldn't process our array type
        Dagger.iscompatible_arg(proc::Dagger.ThreadProc, opts, x::$T) = false

        # Adapt to/from the appropriate type
        function Dagger.move(from_proc::OSProc, to_proc::$PROC, x::Chunk)
            from_pid = from_proc.pid
            to_pid = Dagger.get_parent(to_proc).pid
            @assert myid() == to_pid
            adapt($T, remotecall_fetch(from_pid, x) do x
                poolget(x.handle)
            end)
        end
        function Dagger.move(from_proc::$PROC, to_proc::OSProc, x::Chunk)
            from_pid = Dagger.get_parent(from_proc).pid
            to_pid = to_proc.pid
            @assert myid() == to_pid
            remotecall_fetch(from_pid, x) do x
                adapt(Array, poolget(x.handle))
            end
        end
    end
end

processor(kind::Symbol) = processor(Val(kind))
processor(::Val) = Dagger.ThreadProc
cancompute(kind::Symbol) = cancompute(Val(kind))
cancompute(::Val) = false

kernel_backend() = kernel_backend(Dagger.Sch.thunk_processor())
kernel_backend(::Dagger.ThreadProc) = CPU()

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("cu.jl")
    end
    @require AMDGPU="21141c5a-9bdb-4563-92ae-f87d6854732e" begin
        include("roc.jl")
    end
end

end
