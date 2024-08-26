module CUDAExt

export CuArrayDeviceProc

import Dagger, DaggerGPU, MemPool
import Dagger: CPURAMMemorySpace, Chunk, unwrap
import MemPool: DRef, poolget
import Distributed: myid, remotecall_fetch
import LinearAlgebra
using KernelAbstractions, Adapt

const CPUProc = Union{Dagger.OSProc,Dagger.ThreadProc}

if isdefined(Base, :get_extension)
    import CUDA
else
    import ..CUDA
end
import CUDA: CuDevice, CuContext, CuStream, CuEvent, CuArray, CUDABackend
import CUDA: devices, attribute, context, context!, stream, stream!
import CUDA: CUBLAS, CUSOLVER

using UUIDs

"Represents a single CUDA GPU device."
struct CuArrayDeviceProc <: Dagger.Processor
    owner::Int
    device::Int
    device_uuid::UUID
end
Dagger.get_parent(proc::CuArrayDeviceProc) = Dagger.OSProc(proc.owner)
Dagger.root_worker_id(proc::CuArrayDeviceProc) = proc.owner
Base.show(io::IO, proc::CuArrayDeviceProc) =
    print(io, "CuArrayDeviceProc(worker $(proc.owner), device $(proc.device), uuid $(proc.device_uuid))")
Dagger.short_name(proc::CuArrayDeviceProc) = "W: $(proc.owner), CUDA: $(proc.device)"
DaggerGPU.@gpuproc(CuArrayDeviceProc, CuArray)

"Represents the memory space of a single CUDA GPU's VRAM."
struct CUDAVRAMMemorySpace <: Dagger.MemorySpace
    owner::Int
    device::Int
    device_uuid::UUID
end
Dagger.root_worker_id(space::CUDAVRAMMemorySpace) = space.owner
function Dagger.memory_space(x::CuArray)
    dev = CUDA.device(x)
    device_id = dev.handle
    device_uuid = CUDA.uuid(dev)
    return CUDAVRAMMemorySpace(myid(), device_id, device_uuid)
end
function Dagger.aliasing(x::CuArray{T}) where T
    space = Dagger.memory_space(x)
    S = typeof(space)
    # TODO: Don't switch context, it's wasteful
    ptr = context!(context(x)) do
        Dagger.RemotePtr{Cvoid}(UInt(Base.unsafe_convert(CUDA.CuPtr{T}, x)), space)
    end
    return Dagger.ContiguousAliasing(Dagger.MemorySpan{S}(ptr, sizeof(T)*length(x)))
end

proc_to_space(proc::CuArrayDeviceProc) = CUDAVRAMMemorySpace(proc.owner, proc.device, proc.device_uuid)
space_to_proc(space::CUDAVRAMMemorySpace) = CuArrayDeviceProc(space.owner, space.device, space.device_uuid)
Dagger.memory_spaces(proc::CuArrayDeviceProc) = Set([proc_to_space(proc)])
Dagger.processors(space::CUDAVRAMMemorySpace) = Set([space_to_proc(space)])

Dagger.unsafe_free!(x::CuArray) = CUDA.unsafe_free!(x)

function to_device(proc::CuArrayDeviceProc)
    @assert Dagger.root_worker_id(proc) == myid()
    return DEVICES[proc.device]
end
function to_context(proc::CuArrayDeviceProc)
    @assert Dagger.root_worker_id(proc) == myid()
    return CONTEXTS[proc.device]
end
to_context(handle::Integer) = CONTEXTS[handle]
to_context(dev::CuDevice) = to_context(dev.handle)

function with_context!(handle::Integer)
    context!(CONTEXTS[handle])
    stream!(STREAMS[handle])
end
function with_context!(proc::CuArrayDeviceProc)
    @assert Dagger.root_worker_id(proc) == myid()
    with_context!(proc.device)
end
function with_context!(space::CUDAVRAMMemorySpace)
    @assert Dagger.root_worker_id(space) == myid()
    with_context!(space.device)
end
function with_context(f, x)
    old_ctx = context()
    old_stream = stream()

    with_context!(x)
    try
        f()
    finally
        context!(old_ctx)
        stream!(old_stream)
    end
end

#=
function synchronize_noisy()
    t = (@timed CUDA.synchronize()).time
    bt = backtrace()
    iob = IOBuffer()
    println(iob, "Synchronizing ($t seconds):")
    Base.show_backtrace(iob, bt)
    println(iob)
    seekstart(iob)
    msg = String(take!(iob))
    @info "$msg"
end
=#

function sync_with_context(x::Union{Dagger.Processor,Dagger.MemorySpace})
    if Dagger.root_worker_id(x) == myid()
        with_context(CUDA.synchronize, x)
    else
        # Do nothing, as we have received our value over a serialization
        # boundary, which should synchronize for us
    end
end

function sync_across!(from_space::CUDAVRAMMemorySpace, to_space::CUDAVRAMMemorySpace)
    if Dagger.root_worker_id(from_space) == Dagger.root_worker_id(to_space)
        @assert from_space.device != to_space.device
        @assert Dagger.root_worker_id(from_space) == myid()
        event = with_context(from_space) do
            event = CuEvent()
            CUDA.record(event, STREAMS[from_space.device])
            event
        end
        with_context(to_space) do
            CUDA.wait(event, STREAMS[to_space.device])
        end
    else
        sync_with_context(from_space)
    end
end
sync_across!(from_proc::CuArrayDeviceProc, to_proc::CuArrayDeviceProc) =
    sync_across!(proc_to_space(from_proc), proc_to_space(to_proc))

# Allocations
Dagger.allocate_array_func(::CuArrayDeviceProc, ::typeof(rand)) = CUDA.rand
Dagger.allocate_array_func(::CuArrayDeviceProc, ::typeof(randn)) = CUDA.randn
Dagger.allocate_array_func(::CuArrayDeviceProc, ::typeof(ones)) = CUDA.ones
Dagger.allocate_array_func(::CuArrayDeviceProc, ::typeof(zeros)) = CUDA.zeros
struct AllocateUndef{S} end
(::AllocateUndef{S})(T, dims::Dims{N}) where {S,N} = CuArray{S,N}(undef, dims)
Dagger.allocate_array_func(::CuArrayDeviceProc, ::Dagger.AllocateUndef{S}) where S = AllocateUndef{S}()

# In-place
# N.B. These methods assume that later operations will implicitly or
# explicitly synchronize with their associated stream
function Dagger.move!(to_space::Dagger.CPURAMMemorySpace, from_space::CUDAVRAMMemorySpace, to::AbstractArray{T,N}, from::AbstractArray{T,N}) where {T,N}
    if Dagger.root_worker_id(from_space) == myid()
        sync_with_context(from_space)
        with_context!(from_space)
    end
    copyto!(to, from)
    # N.B. DtoH will synchronize
    return
end
function Dagger.move!(to_space::CUDAVRAMMemorySpace, from_space::Dagger.CPURAMMemorySpace, to::AbstractArray{T,N}, from::AbstractArray{T,N}) where {T,N}
    with_context!(to_space)
    copyto!(to, from)
    return
end
function Dagger.move!(to_space::CUDAVRAMMemorySpace, from_space::CUDAVRAMMemorySpace, to::AbstractArray{T,N}, from::AbstractArray{T,N}) where {T,N}
    sync_across!(from_space, to_space)
    with_context!(to_space)
    copyto!(to, from)
    return
end

is_non_gpu(::Function) = true
is_non_gpu(::Type) = true
is_non_gpu(::Symbol) = true
is_non_gpu(::String) = true
function is_non_gpu(x::T) where T
    isbits(x) && return true
    isprimitivetype(T) && return true
    return false
end

# Out-of-place HtoD
function Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x)
    is_non_gpu(x) && return x
    with_context(to_proc) do
        arr = adapt(CuArray, x)
        CUDA.synchronize()
        return arr
    end
end
function Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x::Chunk)
    from_w = Dagger.root_worker_id(from_proc)
    to_w = Dagger.root_worker_id(to_proc)
    @assert myid() == to_w
    cpu_data = remotecall_fetch(unwrap, from_w, x)
    is_non_gpu(cpu_data) && return cpu_data
    with_context(to_proc) do
        arr = adapt(CuArray, cpu_data)
        CUDA.synchronize()
        return arr
    end
end
function Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x::CuArray)
    if CUDA.device(x) == to_device(to_proc)
        return x
    end
    with_context(to_proc) do
        _x = similar(x)
        copyto!(_x, x)
        CUDA.synchronize()
        return _x
    end
end

# Out-of-place DtoH
function Dagger.move(from_proc::CuArrayDeviceProc, to_proc::CPUProc, x)
    is_non_gpu(x) && return x
    with_context(from_proc) do
        CUDA.synchronize()
        _x = adapt(Array, x)
        CUDA.synchronize()
        return _x
    end
end
function Dagger.move(from_proc::CuArrayDeviceProc, to_proc::CPUProc, x::Chunk)
    from_w = Dagger.root_worker_id(from_proc)
    to_w = Dagger.root_worker_id(to_proc)
    @assert myid() == to_w
    remotecall_fetch(from_w, x) do x
        arr = unwrap(x)
        return Dagger.move(from_proc, to_proc, arr)
    end
end
function Dagger.move(from_proc::CuArrayDeviceProc, to_proc::CPUProc, x::CuArray{T,N}) where {T,N}
    with_context(from_proc) do
        CUDA.synchronize()
        _x = Array{T,N}(undef, size(x))
        copyto!(_x, x)
        CUDA.synchronize()
        return _x
    end
end

function array_tracked(A::CuArray, proc::CuArrayDeviceProc)
    dev_stream = STREAMS[proc.device]
    A_stream = A.data.rc.obj.stream
    if dev_stream == A_stream
        return true
    end
    @warn "Untracked: $dev_stream vs $A_stream"
    return false
end
array_tracked(A::CuArray, space::CUDAVRAMMemorySpace) =
    array_tracked(A, space_to_proc(space))

# Out-of-place DtoD
function Dagger.move(from_proc::CuArrayDeviceProc, to_proc::CuArrayDeviceProc, x::Dagger.Chunk{T}) where T<:CuArray
    if from_proc == to_proc
        # Same process and GPU, no change
        arr = unwrap(x)
        if !array_tracked(arr, from_proc)
            with_context(CUDA.synchronize, from_proc)
        end
        return arr
    elseif Dagger.root_worker_id(from_proc) == Dagger.root_worker_id(to_proc)
        # Same process but different GPUs, use DtoD copy
        from_arr = unwrap(x)
        if !array_tracked(from_arr, from_proc)
            with_context(CUDA.synchronize, from_proc)
        else
            sync_across!(from_proc, to_proc)
        end
        return with_context(to_proc) do
            to_arr = similar(from_arr)
            copyto!(to_arr, from_arr)
            return to_arr
        end
    elseif Dagger.system_uuid(from_proc.owner) == Dagger.system_uuid(to_proc.owner) && from_proc.device_uuid == to_proc.device_uuid
        # Same node, we can use IPC
        ipc_handle, eT, shape = remotecall_fetch(from_proc.owner, x) do x
            arr = unwrap(x)
            ipc_handle_ref = Ref{CUDA.CUipcMemHandle}()
            GC.@preserve arr begin
                CUDA.cuIpcGetMemHandle(ipc_handle_ref, pointer(arr))
            end
            (ipc_handle_ref[], eltype(arr), size(arr))
        end
        r_ptr = Ref{CUDA.CUdeviceptr}()
        CUDA.device!(from_proc.device) do
            CUDA.cuIpcOpenMemHandle(r_ptr, ipc_handle, CUDA.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
        end
        ptr = Base.unsafe_convert(CUDA.CuPtr{eT}, r_ptr[])
        arr = unsafe_wrap(CuArray, ptr, shape; own=false)
        finalizer(arr) do arr
            CUDA.cuIpcCloseMemHandle(pointer(arr))
        end
        if from_proc.device_uuid != to_proc.device_uuid
            return CUDA.device!(to_proc.device) do
                to_arr = similar(arr)
                copyto!(to_arr, arr)
                to_arr
            end
        else
            return arr
        end
    else
        # Different node, use DtoH, serialization, HtoD
        return CuArray(remotecall_fetch(from_proc.owner, x) do x
            Array(unwrap(x))
        end)
    end
end

# Adapt generic functions/types
Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x::Function) = x
Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x::Type) = x
Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x::Chunk{T}) where {T<:Function} =
    Dagger.move(from_proc, to_proc, fetch(x))
Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x::Chunk{T}) where {T<:Type} =
    Dagger.move(from_proc, to_proc, fetch(x))

# Adapt BLAS/LAPACK functions
import LinearAlgebra: BLAS, LAPACK
for lib in [BLAS, LAPACK]
    for name in names(lib; all=true)
        name == nameof(lib) && continue
        startswith(string(name), '#') && continue
        endswith(string(name), '!') || continue

        for culib in [CUBLAS, CUSOLVER]
            if name in names(culib; all=true)
                fn = getproperty(lib, name)
                cufn = getproperty(culib, name)
                @eval Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, ::$(typeof(fn))) = $cufn
            end
        end
    end
end

# Task execution
function Dagger.execute!(proc::CuArrayDeviceProc, f, args...; kwargs...)
    @nospecialize f args kwargs
    tls = Dagger.get_tls()
    task = Threads.@spawn begin
        Dagger.set_tls!(tls)
        with_context!(proc)
        result = Base.@invokelatest f(args...; kwargs...)
        # N.B. Synchronization must be done when accessing result or args
        return result
    end

    try
        fetch(task)
    catch err
        stk = current_exceptions(task)
        err, frames = stk[1]
        rethrow(CapturedException(err, frames))
    end
end

DaggerGPU.processor(::Val{:CUDA}) = CuArrayDeviceProc
DaggerGPU.cancompute(::Val{:CUDA}) = CUDA.has_cuda()
DaggerGPU.kernel_backend(::CuArrayDeviceProc) = CUDABackend()
DaggerGPU.with_device(f, proc::CuArrayDeviceProc) =
    CUDA.device!(f, proc.device)

function Dagger.to_scope(::Val{:cuda_gpu}, sc::NamedTuple)
    if sc.cuda_gpu == Colon()
        return Dagger.to_scope(Val{:cuda_gpus}(), merge(sc, (;cuda_gpus=Colon())))
    else
        @assert sc.cuda_gpu isa Integer "Expected a single GPU device ID for :cuda_gpu, got $(sc.cuda_gpu)\nConsider using :cuda_gpus instead."
        return Dagger.to_scope(Val{:cuda_gpus}(), merge(sc, (;cuda_gpus=[sc.cuda_gpu])))
    end
end
Dagger.scope_key_precedence(::Val{:cuda_gpu}) = 1
function Dagger.to_scope(::Val{:cuda_gpus}, sc::NamedTuple)
    if haskey(sc, :worker)
        workers = Int[sc.worker]
    elseif haskey(sc, :workers) && sc.workers != Colon()
        workers = sc.workers
    else
        workers = map(gproc->gproc.pid, Dagger.procs(Dagger.Sch.eager_context()))
    end
    scopes = Dagger.ExactScope[]
    dev_ids = sc.cuda_gpus
    for worker in workers
        procs = Dagger.get_processors(Dagger.OSProc(worker))
        for proc in procs
            proc isa CuArrayDeviceProc || continue
            if dev_ids == Colon() || proc.device+1 in dev_ids
                scope = Dagger.ExactScope(proc)
                push!(scopes, scope)
            end
        end
    end
    return Dagger.UnionScope(scopes)
end
Dagger.scope_key_precedence(::Val{:cuda_gpus}) = 1

const DEVICES = Dict{Int, CuDevice}()
const CONTEXTS = Dict{Int, CuContext}()
const STREAMS = Dict{Int, CuStream}()

function __init__()
    if CUDA.has_cuda()
        for dev in CUDA.devices()
            @debug "Registering CUDA GPU processor with Dagger: $dev"
            Dagger.add_processor_callback!("cuarray_device_$(dev.handle)") do
                proc = CuArrayDeviceProc(myid(), dev.handle, CUDA.uuid(dev))
                DEVICES[dev.handle] = dev
                ctx = context(dev)
                CONTEXTS[dev.handle] = ctx
                context!(ctx) do
                    STREAMS[dev.handle] = stream()
                end
                return proc
            end
        end
    end
end

end # module CUDAExt
