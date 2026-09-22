import os

import pytest

from your.candidate import Candidate
from your.utils.gpu import *
from your.utils.misc import crop

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
_install_dir = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_gpu_dedisperse():
    file = os.path.join(_install_dir, "data/28.fil")
    cand = Candidate(
        fp=file,
        dm=475.28400,
        tcand=2.0288800,
        width=2,
        label=-1,
        snr=16.8128,
        min_samp=256,
        device=0,
    )
    cand.get_chunk()
    cand.dedisperse(target="GPU")
    g_dedisp = cand.dedispersed
    cand.dedisperse(target="CPU")
    c_dedisp = cand.dedispersed
    assert np.isclose(np.mean(g_dedisp - c_dedisp), 0, atol=1)
    assert np.isclose(np.max(cand.dedispersed.T.sum(0)), 47527, atol=1)


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_gpu_dmt():
    file = os.path.join(_install_dir, "data/28.fil")
    cand = Candidate(
        fp=file,
        dm=10,
        tcand=2.0288800,
        width=2,
        label=-1,
        snr=16.8128,
        min_samp=256,
        device=0,
    )
    cand.get_chunk()
    cand.dmtime(target="GPU")
    g_dmt = cand.dmt
    cand.dmtime(target="CPU")
    c_dmt = cand.dmt
    assert cand.dmt.shape[0] == 256
    assert np.isclose(np.mean(g_dmt - c_dmt), 0, atol=1)
    assert np.max(g_dmt - c_dmt) / np.max(g_dmt) < 0.05


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_gpu_dedisp_dmt_crop():
    file = os.path.join(_install_dir, "data/28.fil")
    cand = Candidate(
        fp=file,
        dm=10,
        tcand=2.0288800,
        width=2,
        label=-1,
        snr=16.8128,
        min_samp=256,
        device=0,
    )
    cand.get_chunk()
    cand = gpu_dedisp_and_dmt_crop(cand)
    g_dmt = cand.dmt
    g_dedisp = cand.dedispersed
    assert cand.dedispersed.shape[0] == 256
    assert cand.dmt.shape[1] == 256

    cand.dedisperse()
    cand.dmtime()
    crop_start_sample_ft = cand.dedispersed.shape[0] // 2 - 256 // 2
    crop_start_sample_dmt = cand.dmt.shape[1] // 2 - 256 // 2
    c_dmt = crop(cand.dmt, crop_start_sample_dmt, 256, 1)
    c_dedisp = crop(cand.dedispersed, crop_start_sample_ft, 256, 0)

    assert np.isclose(np.sum(g_dmt - c_dmt), 0, atol=1)
    assert np.isclose(np.sum(g_dedisp - c_dedisp), 0, atol=1)


def _dmt_cand(dm):
    file = os.path.join(_install_dir, "data/28.fil")
    cand = Candidate(
        fp=file,
        dm=dm,
        tcand=2.0288800,
        width=2,
        label=-1,
        snr=16.8128,
        min_samp=256,
        device=0,
    )
    cand.get_chunk()
    return cand


# dm 10 collapses the band to 0.03 spans per channel, dm 475 to 0.82, so the
# two sit either side of the default crossover and take a kernel each
@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dm", [10, 475.284])
def test_gpu_dmt_kernels_agree(dm):
    """Which kernel runs is a speed choice, so it must not change the plane."""
    runs = gpu_dmt(_dmt_cand(dm), max_run_fraction=1.0).dmt
    channels = gpu_dmt(_dmt_cand(dm), max_run_fraction=0.0).dmt
    assert np.array_equal(runs, channels)


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dm", [10, 475.284])
def test_gpu_dmt_matches_cpu(dm):
    cand = _dmt_cand(dm)
    cand.dmtime(target="CPU")
    cpu = cand.dmt.copy()
    assert np.array_equal(gpu_dmt(_dmt_cand(dm)).dmt, cpu)


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_gpu_dmt_picks_each_kernel():
    for dm, expected in ((10, True), (475.284, False)):
        cand = _dmt_cand(dm)
        freqs = np.asarray(cand.chan_freqs, dtype=np.float64)
        dms = cand.dm + np.linspace(-cand.dm, cand.dm, 256)
        _, nruns = run_edges(delay_table(freqs, float(cand.your_header.tsamp), dms))
        fraction = nruns.mean() / len(freqs)
        assert bool(fraction <= 0.6) is expected
