import os
import json
import h5py
import h5ify
from bilby.gw.prior import UniformSourceFrame


def main():
    os.system(
        'wget "https://gwosc.org/eventapi/json/query/show?' \
        'release=GWTC-2.1-confident,GWTC-3-confident&min-mass-2-source=3&max-far=1" ' \
        '-O events.json',
    )

    events = json.load(open('events.json', 'r'))['events']

    for key in sorted(events):
        name = events[key]['commonName']

        if not os.path.exists(f'{name}.h5'):
            url = events[key]['jsonurl']
            os.system(f'wget {url} -O {name}.json')
            event = json.load(open(f'{name}.json', 'r'))['events'][key]
            pe = sorted([pe for pe in event['parameters'] if 'pe' in pe])[-1]
            url = event['parameters'][pe]['data_url']
            assert '_cosmo' in url
            os.system(f'wget {url} -O {name}.h5')

        if not os.path.exists('pe.h5') or name not in h5py.File('pe.h5', 'r'):
            with h5py.File(f'{name}.h5', 'r') as f:
                d = f['C01:Mixed']['posterior_samples'][:]

            posterior = {par: d[par] for par in (
                'mass_1_source', 'mass_ratio', 'redshift',
                'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2',
            )}

            posterior['prior'] = UniformSourceFrame(
                minimum = posterior['redshift'].min(),
                maximum = posterior['redshift'].max(),
                cosmology = 'Planck15_LAL',
                name = 'redshift',
            ).prob(posterior['redshift'])

            h5ify.save(
                'pe.h5', {name: posterior}, mode = 'a',
                compression = 'gzip', compression_opts = 9,
            )

    return


if __name__ == '__main__':
    main()
