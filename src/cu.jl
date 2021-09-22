using .CUDA
import .CUDA: CuDevice, CuContext, devices, attribute

using UUIDs

export CuArrayDeviceProc

"Represents a single CUDA GPU device."
struct CuArrayDeviceProc <: Dagger.Processor
    owner::Int
    device::Int
    device_uuid::UUID
end
@gpuproc(CuArrayDeviceProc, CuArray)
Dagger.get_parent(proc::CuArrayDeviceProc) = Dagger.OSProc(proc.owner)

# function can_access(this, peer)
#     status = Ref{Cint}()
#     CUDA.cuDeviceCanAccessPeer(status, this, peer)
#     return status[] == 1
# end

function Dagger.move(from::CuArrayDeviceProc, to::CuArrayDeviceProc, x::Dagger.Chunk{T}) where T<:CuArray
    if from == to
        # Same process and GPU, no change
        poolget(x.handle)
    elseif from.owner == to.owner
        # Same process but different GPUs, use DtoD copy
        from_arr = poolget(x.handle)
        to_arr = CUDA.device!(to.device) do
            CuArray{T,N}(undef, size)
        end
        copyto!(to_arr, from_arr)
        to_arr
    elseif Dagger.system_uuid(from.owner) == Dagger.system_uuid(to.owner)
        # Same node, we can use IPC
        ipc_handle, eT, shape = remotecall_fetch(from.owner, x.handle) do h
            arr = poolget(h)
            ipc_handle_ref = Ref{CUDA.CUipcMemHandle}()
            GC.@preserve arr begin
                CUDA.cuIpcGetMemHandle(ipc_handle_ref, pointer(arr))
            end
            (ipc_handle_ref[], eltype(arr), size(arr))
        end
        r_ptr = Ref{CUDA.CUdeviceptr}()
        CUDA.device!(from.device) do # FIXME: Assumes that device IDs are identical across processes
            CUDA.cuIpcOpenMemHandle(r_ptr, ipc_handle, CUDA.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
        end
        ptr = Base.unsafe_convert(CUDA.CuPtr{eT}, r_ptr[])
        arr = unsafe_wrap(CuArray, ptr, shape; own=false)
        finalizer(arr) do arr
            CUDA.cuIpcCloseMemHandle(pointer(arr))
        end
        if from.device_uuid != to.device_uuid
            CUDA.device!(to.device) do
                to_arr = similar(arr)
                copyto!(to_arr, arr)
                to_arr
            end
        else
            arr
        end
    else
        # Different node, use DtoH, serialization, HtoD
        # TODO UCX
        CuArray(remotecall_fetch(from.owner, x.handle) do h
            Array(poolget(h))
        end)
    end
end

function Dagger.execute!(proc::CuArrayDeviceProc, func, args...)
    tls = Dagger.get_tls()
    task = Threads.@spawn begin
        Dagger.set_tls!(tls)
        CUDA.device!(proc.device)
        CUDA.@sync func(args...)
    end
    try
        fetch(task)
    catch err
        @static if VERSION >= v"1.1"
            stk = Base.catch_stack(task)
            err, frames = stk[1]
            rethrow(CapturedException(err, frames))
        else
            rethrow(task.result)
        end
    end
end
Base.show(io::IO, proc::CuArrayDeviceProc) =
    print(io, "CuArrayDeviceProc on worker $(proc.owner), device $(proc.device), uuid $(proc.device_uuid)")

processor(::Val{:CUDA}) = CuArrayDeviceProc
cancompute(::Val{:CUDA}) = CUDA.has_cuda()
kernel_backend(::CuArrayDeviceProc) = CUDADevice()

if CUDA.has_cuda()
    for dev in devices()
        Dagger.add_processor_callback!("cuarray_device_$(dev.handle)") do
            CuArrayDeviceProc(Distributed.myid(), dev.handle, CUDA.uuid(dev))
        end
    end
end
