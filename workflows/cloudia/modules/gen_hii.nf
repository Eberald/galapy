// Author: Enrico Veraldi
// nextyflow module for generation HII input files (cloudy)

process GEN_INPUT_HII {
    /**
     * Generates Cloudy input deck files (.in) and staging SED files for HII region
     * photoionization simulations using the grid specifications and extracted SSP SEDs.
     *
     * Resource Requirements:
     *   - CPUs: 2
     *   - Memory: 2 GB
     *   - Time: 10 minutes
     *
     * Inputs:
     *   - spec_file: Path to HDF5 file containing job specifications (e.g., hii_grid_spec.h5)
     *   - sed_dir: Directory containing the extracted .sed files
     *
     * Outputs:
     *   - decks: Generated Cloudy input files (hii_*.in)
     *   - manifest: Manifest file (jobs.txt) listing job IDs
     *   - seds_staging: Staged SED files directory (SED) required for the runs
     *
     * Published to: ../../../data/input_hii/
     */

    cpus 2
    memory '2 GB'
    time '10 min'
    publishDir '../../../data/input_hii/', mode: 'move'

    input:
    path spec_file     // file HDF5 generated (es. hii_grid_spec.h5)
    path sed_dir       // directory for .sed

    output:
    path 'hii_*.in', emit: decks
    path 'jobs_hii.txt', emit: manifest
    path 'SED',      emit: seds_staging

    script:
    """
    galapy-gen-hii \\
        --spec ${spec_file} \\
        --sed-dir ${sed_dir} \\
        --limit 100 \\
        --out .
    """
}