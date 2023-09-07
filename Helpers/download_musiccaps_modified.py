"""
Download the clips within the MusicCaps dataset from YouTube.

Requires:
    - ffmpeg
    - yt-dlp
    - datasets[audio]
    - torchaudio
"""
import subprocess
import os
from pathlib import Path

from datasets import load_dataset, Audio

MUSIC_TO_SKIP = ['-sevczF5etI', '0J_2K1Gvruk', '0fqtA_ZBn_8', '0khKvVDyYV4', '0pewITE1550', '25Ccp8usBtE', '2dyxjGTXSpA', '374R7te0ra0', '5Y_mT93tkvQ', '63rqIYPHvlc', 
                 '6xxu6f0f0e4', '7B1OAtD_VIA', '7WZwlOrRELI', '8olHAhUKkuk', 'Ah_aYOGnQ_I', 'B7iRvj8y9aU', 'BeFzozm_H5M', 'BiQik0xsWxk', 'C7OIuhWSbjU', 'CCFYOw8keiI', 
                 'EhFWLbNBOxc', 'Fv9swdLA-lo', 'HAHn_zB47ig', 'Hvs6Xwc6-gc', 'IbJh1xeBFcI', 'JNw0A8pRnsQ', 'Jk2mvFrdZTU', 'L5Uu_0xEZg4', 'LRfVQsnaVQE', 'MYtq46rNsCA', 
                 'NIcsJ8sEd0M', 'NXuB3ZEpM5U', 'OS4YFp3DiEE', 'RQ0-sjpAPKU', 'Rqv5fu1PCXA', 'SLq-Co_szYo', 'T6iv9GFIVyU', 'TkclVqlyKx4', 'UdA6I_tXVHE', 'Vu7ZUUl4VPc', 
                 'We0WIPYrtRE', 'WvEtOYCShfM', 'Xoke1wUwEXY', 'Xy7KtmHMQzU', 'ZZrvO__SNtA', '_ACyiKGpD8Y', '_DHMdtRRJzE', '_hYBs0xee9Y', 'asYb6iDz_kM', 'cADT8fUucLQ', 
                 'd6-bQMCz7j0', 'dcY062mkf9g', 'eHeUipPZHIc', 'ed-1zAOr9PQ', 'fZyq2pM2-dI', 'fwXh_lMOqu0', 'g8USMvt9np0', 'gdtw54I8soM', 'go_7i6WvfeE', 'iXgEQj1Fs7g', 
                 'idVCpQaByc4', 'j9hAUlz5kQs', 'jd1IS7N3u0I', 'jmPmqzxlOTY', 'k-LkhT4HAiE', 'kiu-40_T5nY', 'lTLsL94ABRs', 'lrk00BNiuD4', 'm-e3w2yZ6sM', 'nTtxF9Wyw6o',
                 'p_-lKpxLK3g', 'qc1DaM4kdO0', 't5fW1-6iXZY', 'tpamd6BKYU4', 'vOAXAoHtl7o', 'vQHKa69Mkzo', 'vVNWjq9byoQ', 'xxCnmao8FAs', 'zCrpaLEq1VQ', 'zSSIGv82318', 
                 'zSq2D_GF00o']

def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir='C:\\Projects\\GMM\\tmp',
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v='
):
    status = False

    command = f"""
        yt-dlp --quiet --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" "{url_base}{video_identifier}"
    """.strip()

    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def main(
    data_dir: str,
    sampling_rate: int = 44100,
    limit: int = None,
    num_proc: int = 1,
    writer_batch_size: int = 1000,
):
    """
    Download the clips within the MusicCaps dataset from YouTube.

    Args:
        data_dir: Directory to save the clips to.
        sampling_rate: Sampling rate of the audio clips.
        limit: Limit the number of examples to download.
        num_proc: Number of processes to use for downloading.
        writer_batch_size: Batch size for writing the dataset. This is per process.
    """

    ds = load_dataset('google/MusicCaps', split='train')
    ds = ds.filter(lambda example: example['ytid'] not in MUSIC_TO_SKIP)

    if limit is not None:
        print(f"Limiting to {limit} examples")
        ds = ds.select(range(limit))

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        print(outfile_path)
        status = True
        if not os.path.exists(outfile_path):
            status = False
            status, log = download_clip(
                example['ytid'],
                outfile_path,
                example['start_s'],
                example['end_s'],
            )
            if not status:
                print(f"Error downloading clip {example['ytid']}: {log}")

        example['audio'] = outfile_path
        example['download_status'] = status
        return example

    return ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False
    ).cast_column('audio', Audio(sampling_rate=sampling_rate))


if __name__ == '__main__':
    ds = main(
        './music_data',
        sampling_rate=44100,
        limit=None,
        num_proc=16,
        writer_batch_size=1000,
    )