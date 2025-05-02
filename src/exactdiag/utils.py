import logging
import pathlib
from os.path import isfile


_logger = logging.getLogger(__name__)


def load_calculate_save(*paths: tuple[pathlib.Path, ...], load_fun, calc_fun, save_fun):
    already_exist = all(isfile(path) for path in paths)
    str_paths = [str(p) for p in paths]
    if already_exist:
        try:
            result = load_fun(*paths)
            _logger.debug(f"loaded {str_paths}")
        except OSError as e:
            message = f"Loading {str_paths}: file(s) exists but load failed. {e}"
            _logger.debug(message)
            raise OSError(message)
    else:
        _logger.debug(f"starting calculating {str_paths}")
        result = calc_fun()
        save_fun(result, *paths)
        _logger.debug(f"finished calculating {str_paths}")
    return result


def set_mkl_threads(no_cores):
    # set number of cores used by scipy - different procedures use different settings?
    try:
        from mkl import set_num_threads

        set_num_threads(no_cores)
        return 0
    except Exception:
        pass
    for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
        try:
            import ctypes

            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(no_cores)))
            return 0
        except Exception:
            pass
    from os import environ

    environ["OMP_NUM_THREADS"] = str(no_cores)  # export OMP_NUM_THREADS
    environ["OMP_PROC_BIND"] = "spread"  # also sets thread affinity, i think
    environ["OPENBLAS_NUM_THREADS"] = str(no_cores)
    environ["MKL_NUM_THREADS"] = str(no_cores)
    environ["VECLIB_MAXIMUM_THREADS"] = str(no_cores)
    environ["NUMEXPR_NUM_THREADS"] = str(no_cores)
