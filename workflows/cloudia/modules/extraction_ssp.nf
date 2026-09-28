// Author: Enrico Veraldi
// nextyflow module for extraction of SSP from galapy lib to table SED for Cloudy

process EXTRACT_SSP_SEDS{
    /**
     * Extracts Simple Stellar Population (SSP) Spectral Energy Distributions (SEDs)
     * from the galapy library and converts them to Cloudy table format (.sed files).
     *
     * This process generates individual SED files for different stellar population
     * ages and metallicities from the specified SSP library (default: parsec22.NT).
     * The extracted SEDs are written in the task directory along with the metadata
     * catalog in JSON format (galapy-sed-cloudy-extract).
     *
     * Resource Requirements:
     *   - CPUs: 1
     *   - Memory: 2 GB
     *   - Time: 2 minutes
     *
     * Outputs:
     *   - seds: Collection of .sed files (ssp_tau*_Z*.sed), one per SSP node
     *   - meta: JSON file (ssp_metadata.json) listing the extracted SSP nodes
     *           (tau_SSP, Z_star, sed_file, it, iz, Qh_unit)
     *
     * Published to: ../../../data/cloudy_seds/
     */
    cpus 1
    memory '2 GB'
    time '2m'
    publishDir '../../../data/cloudy_seds/', mode: 'copy'

    output:
    path '*.sed', emit: seds
    path 'ssp_metadata.json', emit: meta

    script:
    """
    galapy-sed-cloudy-extract -o .
    """
}
