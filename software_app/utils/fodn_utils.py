# file: utils/fodn_utils.py

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from .fodn_code import fracOrdUU

def _process_single_chunk(args):
    """
    A helper function that runs fracOrdUU on a single chunk.
    This function must be at top-level for ProcessPoolExecutor to work on Windows.
    """
    chunk_data, numFract, niter, lambdaUse = args
    model = fracOrdUU(numFract=numFract, niter=niter, lambdaUse=lambdaUse, verbose=0)
    model._numCh = chunk_data.shape[0]
    model._K = chunk_data.shape[1]
    model.fit(chunk_data)
    return model._order, model._AMat[-1, :, :]

def run_fodn_analysis(data, chunk_size=1.0, numFract=50, niter=10, lambdaUse=0.5, base_time=0.0, max_workers=4):
    """
    Processes the input data chunk by chunk in parallel using ProcessPoolExecutor.
    For each complete chunk, we run the fracOrdUU fit() and return results
    (alpha values, coupling matrix) along with chunk start/end times.

    Parameters:
      data: 2D numpy array (channels, timepoints)
      chunk_size: duration (in seconds) of each chunk
      numFract, niter, lambdaUse: FODN parameters
      base_time: absolute start time offset for labeling chunk times
      max_workers: number of processes to use for parallelization

    Returns:
      A list of dicts. Each dict has:
        "alpha": per-channel fractional exponents
        "coupling_matrix": final coupling matrix
        "chunk_start": absolute start time for the chunk
        "chunk_end": absolute end time for the chunk
    """
    fs = 1000  # assumed sampling rate
    samples_per_chunk = int(chunk_size * fs)
    num_chunks = data.shape[1] // samples_per_chunk

    chunk_args = []
    chunk_info = [] 
    for i in range(num_chunks):
        start_idx = i * samples_per_chunk
        end_idx = (i + 1) * samples_per_chunk
        chunk_data = data[:, start_idx:end_idx]

        chunk_args.append((chunk_data, numFract, niter, lambdaUse))

        c_start = base_time + i * chunk_size
        c_end = base_time + (i + 1) * chunk_size
        chunk_info.append((i, c_start, c_end))

    results = [None] * num_chunks 

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = {}
        for i, args in enumerate(chunk_args):
            fut = executor.submit(_process_single_chunk, args)
            futures[fut] = i


        for fut in as_completed(futures):
            idx = futures[fut]
            alpha_vals, coupling_mat = fut.result()
            chunk_idx, c_start, c_end = chunk_info[idx]
            chunk_dict = {
                "alpha": alpha_vals,
                "coupling_matrix": coupling_mat,
                "chunk_start": c_start,
                "chunk_end": c_end
            }
            results[idx] = chunk_dict

    print(f"Parallel FODN computation took {time.time() - t0:.2f} seconds using up to {max_workers} workers.")
    return results

