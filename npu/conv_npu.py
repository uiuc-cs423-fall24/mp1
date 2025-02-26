import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_w, filter_height, filter_width = W.shape
    out_channels_b = bias.shape[0]

    out_height = input_height - filter_height + 1
    out_width  = input_width  - filter_width + 1

    # Prepare the output shape
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm
    )

    TILE_K = 128    
    TILE_M = 128      
    TILE_N = 512       


    ohw = out_height * out_width

    for b in nl.affine_range(batch_size):

        # Zero out the entire accumulator in HBM:
        # shape: (out_channels, ohw)
        out_accum = nl.zeros(
            shape=(out_channels, ohw),
            dtype=X.dtype,
            buffer=nl.hbm
        )

        # Slide over all filter offsets (i, j)
        for fi in nl.affine_range(filter_height):
            for fj in nl.affine_range(filter_width):


                K_tiles = in_channels // TILE_K
                M_tiles = out_channels // TILE_M
                N_tiles = (ohw + (TILE_N - 1)) // TILE_N  

                for mt in nl.affine_range(M_tiles):
                    # out-channels chunk
                    oc_begin = mt * TILE_M
                    oc_end   = oc_begin + TILE_M

                    for nt in nl.affine_range(N_tiles):
                        n_begin = nt * TILE_N
                        n_size  = nl.min(TILE_N, ohw - n_begin)
                        n_end   = n_begin + n_size

                        # Load current partial-sum tile from HBM:
                        old_sum_tile = nl.ndarray((TILE_M, n_size), dtype=X.dtype, buffer=nl.sbuf)
                        old_sum_tile[...] = nl.load(out_accum[oc_begin:oc_end, n_begin:n_end])

                        # We'll accumulate into PSUM
                        psum_tile = nl.copy(old_sum_tile, dtype=X.dtype, buffer=nl.psum)

                        # Loop over the K dimension in tile chunks
                        for kt in nl.affine_range(K_tiles):
                            k_begin = kt * TILE_K
                            k_end   = k_begin + TILE_K

                            # Load W-tile of shape (TILE_K, TILE_M):

                            w_tile = nl.ndarray((TILE_K, TILE_M), dtype=W.dtype, buffer=nl.sbuf)
                            w_tile[...] = nl.load(W[oc_begin:oc_end, k_begin:k_end, fi, fj], transpose_indices=True)

                            x_tile = nl.ndarray((TILE_K, n_size), dtype=X.dtype, buffer=nl.sbuf)

                            for c_ in nl.affine_range(TILE_K):
                                c_real = k_begin + c_
                                for n_ in nl.affine_range(n_size):
                                    out_hw = n_begin + n_
                                    oh = out_hw // out_width
                                    ow = out_hw %  out_width
                                    x_tile[c_, n_] = nl.load(X[b, c_real, oh+fi, ow+fj])

                            partial = nl.matmul(w_tile, x_tile, transpose_x=False, transpose_y=False)
                            # Accumulate partial into psum_tile:
                            psum_tile[...] = nl.add(psum_tile, partial)

                        nl.store(out_accum[oc_begin:oc_end, n_begin:n_end], value=psum_tile)

        # add bias for each output channel. We again tile over (out_channels, ohw).
        M_tiles = out_channels // TILE_M
        N_tiles = (ohw + (TILE_N - 1)) // TILE_N

        for mt in nl.affine_range(M_tiles):
            oc_begin = mt * TILE_M
            oc_end   = oc_begin + TILE_M

            for nt in nl.affine_range(N_tiles):
                n_begin = nt * TILE_N
                n_size  = nl.min(TILE_N, ohw - n_begin)
                n_end   = n_begin + n_size

                accum_tile = nl.ndarray((TILE_M, n_size), dtype=X.dtype, buffer=nl.sbuf)
                accum_tile[...] = nl.load(out_accum[oc_begin:oc_end, n_begin:n_end])

                # add the bias in a vectorized manner. 
                bias_tile = nl.load(bias[oc_begin:oc_end])  # shape (TILE_M,)
                for r_ in nl.affine_range(TILE_M):
                    for col_ in nl.affine_range(n_size):
                        accum_tile[r_, col_] = nl.add(accum_tile[r_, col_], bias_tile[r_])

                nl.store(out_accum[oc_begin:oc_end, n_begin:n_end], value=accum_tile)


        out_accum_reshaped = nl.ndarray(
            shape=(out_channels, out_height, out_width),
            dtype=X.dtype,
            buffer=nl.sbuf
        )

        for oh_ in nl.affine_range(out_height):

            row_tile = nl.copy(out_accum[:, oh_*out_width:(oh_+1)*out_width], dtype=X.dtype)
            out_accum_reshaped[:, oh_, :] = row_tile

        # Store into X_out[b]
        nl.store(X_out[b], value=out_accum_reshaped)

    return X_out
