import logging
import math
import subprocess

import numpy as np
from numba import cuda

logger = logging.getLogger(__name__)


def gpu_dedisperse(cand, device=0):
    """

    GPU dedispersion (by rolling the array)

    Args:
        cand: Candidate instance
        device (int): GPU ID

    Returns:
        candidate object

    """
    cuda.select_device(device)
    chan_freqs = cuda.to_device(np.array(cand.chan_freqs, dtype=np.float32))
    cand_data_in = cuda.to_device(np.array(cand.data.T, dtype=cand.your_header.dtype))
    cand_data_out = cuda.to_device(
        np.zeros_like(cand.data.T, dtype=cand.your_header.dtype)
    )

    @cuda.jit
    def gpu_dedisp(cand_data_in, chan_freqs, dm, cand_data_out, tsamp):
        ii, jj = cuda.grid(2)
        if ii < cand_data_in.shape[0] and jj < cand_data_in.shape[1]:
            disp_time = int(
                round(
                    -4148808.0
                    * dm
                    * (1 / (chan_freqs[0]) ** 2 - 1 / (chan_freqs[ii]) ** 2)
                    / 1000
                    / tsamp
                )
            )
            cand_data_out[ii, jj] = cand_data_in[
                ii, (jj + disp_time) % cand_data_in.shape[1]
            ]

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(cand_data_in.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(cand_data_in.shape[1] / threadsperblock[1])

    blockspergrid = (blockspergrid_x, blockspergrid_y)

    gpu_dedisp[blockspergrid, threadsperblock](
        cand_data_in,
        chan_freqs,
        float(cand.dm),
        cand_data_out,
        float(cand.your_header.tsamp),
    )

    cand.dedispersed = cand_data_out.copy_to_host().T

    return cand


@cuda.jit
def dmt_channels(cand_data_in, delays, cand_data_out):
    """
    One thread per (DM, sample), summing the channels in a register.

    Args:
        cand_data_in: (nchans, nsamples) chunk, channel major
        delays: (ndm, nchans) sample shift per DM and channel
        cand_data_out: (ndm, nsamples) DM-time plane

    """
    jj, kk = cuda.grid(2)
    nsamples = cand_data_in.shape[1]
    if jj < nsamples and kk < cand_data_out.shape[0]:
        acc = 0
        for ii in range(cand_data_in.shape[0]):
            # a negative shift wraps, Python modulo semantics, as numba gives
            acc += cand_data_in[ii, (jj + delays[kk, ii]) % nsamples]
        cand_data_out[kk, jj] = acc


@cuda.jit
def dmt_runs(cumsum, starts, stops, delays, nruns, cand_data_out):
    """
    As `dmt_channels`, but each span of channels sharing a delay bin is one
    subtraction of the band cumulative sum rather than a loop over its channels.

    Args:
        cumsum: (nchans + 1, nsamples) cumulative sum down the channel axis
        starts: (ndm, nruns) first channel of each span
        stops: (ndm, nruns) one past the last channel of each span
        delays: (ndm, nruns) sample shift of each span
        nruns: (ndm,) spans this DM actually has, the rest being padding
        cand_data_out: (ndm, nsamples) DM-time plane

    """
    jj, kk = cuda.grid(2)
    nsamples = cumsum.shape[1]
    if jj < nsamples and kk < cand_data_out.shape[0]:
        acc = 0
        for rr in range(nruns[kk]):
            idx = (jj + delays[kk, rr]) % nsamples
            acc += cumsum[stops[kk, rr], idx] - cumsum[starts[kk, rr], idx]
        cand_data_out[kk, jj] = acc


@cuda.jit
def band_cumsum(cand_data_in, cumsum):
    """
    Cumulative sum down the channel axis, one thread per sample, with a leading
    zero row so any contiguous span of channels sums as one subtraction.

    Args:
        cand_data_in: (nchans, nsamples) chunk, channel major
        cumsum: (nchans + 1, nsamples) output, row 0 already zeroed

    """
    jj = cuda.grid(1)
    if jj < cand_data_in.shape[1]:
        acc = 0
        for ii in range(cand_data_in.shape[0]):
            acc += cand_data_in[ii, jj]
            cumsum[ii + 1, jj] = acc


def delay_table(chan_freqs, tsamp, dms):
    """
    Sample shift per DM and channel. float64, so it rounds the way the CPU path
    does; the delays used to be worked out in the kernel in float32, where a few
    channels per plane land in a neighbouring bin.

    Args:
        chan_freqs (numpy.ndarray): channel frequencies, MHz
        tsamp (float): sampling time, seconds
        dms (numpy.ndarray): DMs to make the plane over

    Returns:
        numpy.ndarray: (ndm, nchans) shift in samples

    """
    delay = (
        4148808.0
        * dms[:, None]
        * (1 / chan_freqs[0] ** 2 - 1 / chan_freqs[None, :] ** 2)
        / 1000
        / tsamp
    )
    return -np.round(delay).astype(np.int32)


def run_edges(delays):
    """
    Where each span of equal delay starts, and how many spans each DM has.

    Args:
        delays (numpy.ndarray): (ndm, nchans) shift in samples

    Returns:
        tuple: (ndm, nchans) bool of span starts, (ndm,) span counts

    """
    edge = np.empty(delays.shape, dtype=bool)
    edge[:, 0] = True
    np.not_equal(delays[:, 1:], delays[:, :-1], out=edge[:, 1:])
    return edge, edge.sum(axis=1).astype(np.int32)


def run_table(delays, edge, nruns):
    """
    Lay the spans out per DM, padded to the DM with the most of them.

    Args:
        delays (numpy.ndarray): (ndm, nchans) shift in samples
        edge (numpy.ndarray): (ndm, nchans) bool of span starts
        nruns (numpy.ndarray): (ndm,) span counts

    Returns:
        tuple: starts, stops, span delays, each (ndm, max spans)

    """
    ndm, nchans = delays.shape
    width = int(nruns.max())
    kk, ii = np.nonzero(edge)
    rr = np.arange(kk.size) - np.repeat(np.cumsum(nruns) - nruns, nruns)

    starts = np.zeros((ndm, width), dtype=np.int32)
    run_delays = np.zeros((ndm, width), dtype=np.int32)
    starts[kk, rr] = ii
    run_delays[kk, rr] = delays[kk, ii]

    stops = np.zeros((ndm, width), dtype=np.int32)
    stops[:, :-1] = starts[:, 1:]
    stops[np.arange(ndm), nruns - 1] = nchans
    return starts, stops, run_delays


def gpu_dmt(cand, device=0, dmsteps=256, max_run_fraction=0.6):
    """

    GPU DM-Time bow-tie

    Two kernels, identical output. `dmt_runs` sums each span of channels sharing
    a delay bin as one subtraction of a band cumulative sum, so it wins while the
    band still collapses; once more than `max_run_fraction` of the channels earn
    their own bin it reads four bytes per span where `dmt_channels` reads one
    byte per channel, and loses. The default crossover was measured on a T4.

    Args:
        cand: Candidate instance
        device (int): GPU ID
        dmsteps (int): rows in the DM-time plane
        max_run_fraction (float): spans per channel above which to take
            `dmt_channels`

    Returns:
        candidate object

    """
    cuda.select_device(device)
    nsamples, nchans = cand.data.shape
    tsamp = float(cand.your_header.tsamp)
    chan_freqs = np.asarray(cand.chan_freqs, dtype=np.float64)
    # the same expression the CPU path uses, so the DM axes are bit identical
    dms = cand.dm + np.linspace(-cand.dm, cand.dm, dmsteps)

    delays = delay_table(chan_freqs, tsamp, dms)
    # counting the spans is cheap; only lay them out if we are going to use them
    edge, nruns = run_edges(delays)
    fraction = float(nruns.mean()) / nchans

    cand_data_in = cuda.to_device(np.ascontiguousarray(cand.data.T))
    dmt_return = cuda.device_array((dmsteps, nsamples), dtype=np.float32)
    threads = 128
    blocks = math.ceil(nsamples / threads)

    if fraction <= max_run_fraction:
        logger.debug(f"{fraction:.2f} spans per channel, summing runs")
        # int32 holds the whole band for any real filterbank; widen if it cannot
        info = np.iinfo(cand.data.dtype)
        bound = nchans * max(abs(int(info.min)), int(info.max))
        acc_dtype = np.int32 if bound <= np.iinfo(np.int32).max else np.int64
        starts, stops, run_delays = run_table(delays, edge, nruns)
        cumsum = cuda.device_array((nchans + 1, nsamples), dtype=acc_dtype)
        cumsum[0].copy_to_device(np.zeros(nsamples, dtype=acc_dtype))
        band_cumsum[blocks, threads](cand_data_in, cumsum)
        dmt_runs[(blocks, dmsteps), (threads, 1)](
            cumsum,
            cuda.to_device(starts),
            cuda.to_device(stops),
            cuda.to_device(run_delays),
            cuda.to_device(nruns),
            dmt_return,
        )
    else:
        logger.debug(f"{fraction:.2f} spans per channel, summing channels")
        dmt_channels[(blocks, dmsteps), (threads, 1)](
            cand_data_in, cuda.to_device(delays), dmt_return
        )

    cand.dmt = dmt_return.copy_to_host()

    return cand


def gpu_dedisp_and_dmt_crop(cand, device=0):
    """

    GPU based dedispersion, DM time bow-time plot and crop it to 256x256 shaped arrays (by rolling the array)

    Args:
        cand: Candidate instance
        device (int): GPU ID

    Returns:
        candidate object

    """

    if cand.width < 3:
        time_decimation_factor = 1
    else:
        time_decimation_factor = cand.width // 2

    if cand.data.shape[1] < 256:
        raise IndexError("GPU candmaker will not work if nchans < 256.")

    frequency_decimation_factor = math.floor(cand.data.shape[1] // 256)

    logger.debug(f"Freq decimation factor: {frequency_decimation_factor}")
    logger.debug(f"Time decimation factor: {time_decimation_factor}")

    cuda.select_device(device)
    stream = cuda.stream()

    logger.debug("Created CUDA Stream")

    chan_freqs = cuda.to_device(
        np.array(cand.chan_freqs, dtype=np.float32), stream=stream
    )
    cand_data_in = cuda.to_device(
        np.array(cand.data.T, dtype=cand.your_header.dtype), stream=stream
    )
    dmt_on_device = cuda.device_array(
        (256, int(cand.data.shape[0] // time_decimation_factor)),
        dtype=np.float32,
        stream=stream,
    )
    cand_dedispersed_on_device = cuda.device_array(
        (
            int(cand.data.shape[1] / frequency_decimation_factor),
            int(cand.data.shape[0] // time_decimation_factor),
        ),
        dtype=np.float32,
        stream=stream,
    )
    cand_dedispersed_out = cuda.device_array(
        shape=(int(cand.data.shape[1] / frequency_decimation_factor), 256),
        dtype=np.float32,
        stream=stream,
    )
    dmt_return = cuda.device_array(shape=(256, 256), dtype=np.float32, stream=stream)
    dm_list = cuda.to_device(
        np.linspace(0, 2 * cand.dm, 256, dtype=np.float32), stream=stream
    )

    logger.debug("Allocated arrays on the GPU")

    @cuda.jit
    def crop_time(data_in, data_out, side_stride):
        ii, jj = cuda.grid(2)
        if ii < data_out.shape[0] and jj < data_out.shape[1]:
            data_out[ii, jj] = data_in[ii, jj + side_stride]

    @cuda.jit
    def gpu_dedisp(
        cand_data_in,
        chan_freqs,
        dm,
        cand_data_out,
        tsamp,
        time_decimation_factor,
        frequency_decimation_factor,
    ):
        ii, jj = cuda.grid(2)
        if ii < cand_data_in.shape[0] and jj < cand_data_in.shape[1]:
            disp_time = int(
                round(
                    -4148808.0
                    * dm
                    * (1 / (chan_freqs[0]) ** 2 - 1 / (chan_freqs[ii]) ** 2)
                    / 1000
                    / tsamp
                )
            )
            cuda.atomic.add(
                cand_data_out,
                (
                    int(ii / frequency_decimation_factor),
                    int(jj / time_decimation_factor),
                ),
                cand_data_in[ii, (jj + disp_time) % cand_data_in.shape[1]],
            )

    threadsperblock_2d = (32, 32)
    blockspergrid_x_2d_in = math.ceil(cand_data_in.shape[0] / threadsperblock_2d[0])
    blockspergrid_y_2d_in = math.ceil(cand_data_in.shape[1] / threadsperblock_2d[1])

    blockspergrid_2d_in = (blockspergrid_x_2d_in, blockspergrid_y_2d_in)

    gpu_dedisp[blockspergrid_2d_in, threadsperblock_2d, stream](
        cand_data_in,
        chan_freqs,
        float(cand.dm),
        cand_dedispersed_on_device,
        float(cand.your_header.tsamp),
        int(time_decimation_factor),
        int(frequency_decimation_factor),
    )

    blockspergrid_x_2d_out = math.ceil(
        cand_dedispersed_on_device.shape[0] / threadsperblock_2d[0]
    )
    blockspergrid_y_2d_out = math.ceil(
        cand_dedispersed_on_device.shape[1] / threadsperblock_2d[0]
    )

    blockspergrid_2d_out = (blockspergrid_x_2d_out, blockspergrid_y_2d_out)
    crop_time[blockspergrid_2d_out, threadsperblock_2d, stream](
        cand_dedispersed_on_device,
        cand_dedispersed_out,
        int(int(cand_dedispersed_on_device.shape[1] / 2) - 128),
    )
    cand.dedispersed = cand_dedispersed_out.copy_to_host(stream=stream).T

    logger.debug("cand.dedisersed set!")

    disp_time = np.zeros(shape=(cand_data_in.shape[0], 256), dtype=np.int32)
    for idx, dms in enumerate(np.linspace(0, 2 * cand.dm, 256)):
        disp_time[:, idx] = np.round(
            -1
            * 4148808.0
            * dms
            * (1 / (cand.chan_freqs[0]) ** 2 - 1 / (cand.chan_freqs) ** 2)
            / 1000
            / cand.your_header.tsamp
        )

    all_delays = cuda.to_device(disp_time, stream=stream)

    @cuda.jit
    def gpu_dmt(cand_data_in, all_delays, dms, cand_data_out, time_decimation_factor):
        ii, jj, kk = cuda.grid(3)
        if (
            ii < cand_data_in.shape[0]
            and jj < cand_data_in.shape[1]
            and kk < dms.shape[0]
        ):
            cuda.atomic.add(
                cand_data_out,
                (kk, int(jj / time_decimation_factor)),
                cand_data_in[ii, (jj + all_delays[ii, kk]) % cand_data_in.shape[1]],
            )

    threadsperblock_3d = (1, 32, 32)
    blockspergrid_x = math.ceil(cand_data_in.shape[0] / threadsperblock_3d[0])
    blockspergrid_y = math.ceil(cand_data_in.shape[1] / threadsperblock_3d[1])
    blockspergrid_z = math.ceil(dm_list.shape[0] / threadsperblock_3d[2])

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    gpu_dmt[blockspergrid, threadsperblock_3d, stream](
        cand_data_in, all_delays, dm_list, dmt_on_device, int(time_decimation_factor)
    )

    crop_time[blockspergrid_2d_out, threadsperblock_2d, stream](
        dmt_on_device,
        dmt_return,
        int(int(cand_dedispersed_on_device.shape[1] / 2) - 128),
    )

    cand.dmt = dmt_return.copy_to_host(stream=stream)

    logger.debug("cand.dmt set!")
    return cand


def get_gpu_memory_map(gpu_id):
    """
    Get the current gpu free memory

    Args:
        gpu_id (int): GPU id

    Returns:
        int: amount of free GPU RAM
    """
    cmd_list = [
        "nvidia-smi",
        "-i",
        f"{gpu_id}",
        "--query-gpu=memory.free",
        "--format=csv,nounits,noheader",
    ]
    result = subprocess.check_output(cmd_list)
    return int(result.decode())
