import os
import numpy as np
import h5py
import h5ify


def main():
    if not os.path.exists('endo3_bbhpop-LIGO-T2100113-v12.hdf5'):
        os.system('wget https://zenodo.org/records/7890437/files/endo3_bbhpop-LIGO-T2100113-v12.hdf5')

    injections = {}

    with h5py.File('endo3_bbhpop-LIGO-T2100113-v12.hdf5', 'r') as f:
        d = f['injections']

        injections['time'] = d.attrs['analysis_time_s'][()] / 60 / 60 / 24 / 365.25
        injections['total'] = d.attrs['total_generated'][()]

        ifar = np.max([d[key][:] for key in d if 'ifar' in key], axis = 0)
        found = ifar > 1

        m1 = d['mass1_source'][found]
        m2 = d['mass2_source'][found]
        s1x = d['spin1x'][found]
        s1y = d['spin1y'][found]
        s1z = d['spin1z'][found]
        s2x = d['spin2x'][found]
        s2y = d['spin2y'][found]
        s2z = d['spin2z'][found]
        z = d['redshift'][found]

        injections['prior'] = (
            d['mass1_source_mass2_source_sampling_pdf'][found]
            * d['spin1x_spin1y_spin1z_sampling_pdf'][found]
            * d['spin2x_spin2y_spin2z_sampling_pdf'][found]
            * d['redshift_sampling_pdf'][found]
            / d['mixture_weight'][found] # these are all 1 anyway
        )

    q = m2 / m1
    a1 = (s1x**2 + s1y**2 + s1z**2)**0.5
    a2 = (s2x**2 + s2y**2 + s2z**2)**0.5
    c1 = s1z / a1
    c2 = s2z / a2

    injections['mass_1_source'] = m1
    injections['mass_ratio'] = q
    injections['a_1'] = a1
    injections['a_2'] = a2
    injections['cos_tilt_1'] = c1
    injections['cos_tilt_2'] = c2
    injections['redshift'] = z

    injections['prior'] *= 4 * np.pi**2 * a1**2 * a2**2 * m1

    for k in injections:
        injections[k] = np.atleast_1d(injections[k])

    h5ify.save(
        'vt.h5', injections, mode = 'w',
        compression = 'gzip', compression_opts = 9,
    )

    return


if __name__ == '__main__':
    main()
