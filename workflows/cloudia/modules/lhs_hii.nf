// Author: Enrico Veraldi
// nextyflow module for lhs grid creation HII regions

process GEN_LHS_HII {
    /**
     * Generates Latin Hypercube Sampling (LHS) parameter grid and Cloudy job
     * specifications for HII region photoionization modeling.
     * The grid charter (grid_hii.yaml) is resolved from the GalaPy database.
     *
     * Resource Requirements:
     *   - CPUs: 1
     *   - Memory: 2 GB
     *   - Time: 10 minutes
     *
     * Inputs:
     *   - ssp_meta: Path to the SSP metadata file (ssp_metadata.json, EXTRACT_SSP_SEDS.out.meta)
     *
     * Outputs:
     *   - grid_spec: HDF5 file (hii_grid_spec.h5) containing Cloudy job specifications
     *   - lhs_sample: NumPy array file (lhs_hii.npy) containing the LHS parameter samples
     *
     * Published to: ../../../data/grids/hii/
     */
    cpus 1
    memory '2 GB'
    time '10m'
    publishDir '../../../data/grids/hii/', mode: 'copy'

    input:
    path ssp_meta

    output:
    path 'hii_grid_spec.h5', emit: grid_spec
    path 'lhs_hii.npy',   emit: lhs_sample

    script:
    """
    galapy-gen-lhs-hii --ssp-meta ${ssp_meta} -o .
    """
}
