// Author: Enrico Veraldi
// nextyflow module for extraction of SSP from galapy lib to table SED for Cloudy

process EXTRACT_SSP_SEDS{
    /**
     * Extracts Simple Stellar Population (SSP) Spectral Energy Distributions (SEDs)
     * from the galapy library and converts them to Cloudy table format (.sed files).
     *
     * This process generates individual SED files for different stellar population
     * ages and metallicities from the specified SSP library (default: parsec22.NT).
     * The extracted SEDs are stored in the SED/ directory along with metadata in HDF5 format.
     *
     * Resource Requirements:
     *   - CPUs: 1
     *   - Memory: 2 GB
     *   - Time: 2 minutes
     *
     * Outputs:
     *   - seds: Collection of .sed files in SED/ directory, one per SSP model
     *   - meta: HDF5 file (ssp_metadata.h5) containing metadata about the extracted SSPs
     *
     * Published to: ../../../data/cloudy_seds/
     */
    cpus 1
    memory '2 GB'
    time '2m'
    publishDir '../../../data/cloudy_seds/', mode: 'move'

    output:
    path 'SED/*.sed', emit: seds
    path 'SED/metadata.json', emit: meta

    script:
    """
    mkdir -p SED
    python -m galapy.spectroscopy.utils.spectra \\
        --ssp-lib parsec22.NT \\
        --out SED/
    """
}