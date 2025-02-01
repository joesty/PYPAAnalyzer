import json
import os
from scipy.io import wavfile
from scipy.signal import resample_poly
import numpy as np
from libNS.NSAnalyzer import NSAnalyzer
from libR.RAnalyzer import RAnalyzer
from libF.FAnalyzer import FAnalyzer
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import argparse
chunk_length_sec = 1
sample_rate = 16000
from tqdm import tqdm
FIX_FACTOR = 32786

new_sample_rate = 16000

def PAAnalyzer(xr):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_R = executor.submit(RAnalyzer, xr)
        future_NS = executor.submit(NSAnalyzer, xr)
        future_F = executor.submit(FAnalyzer, xr)

        R = future_R.result()
        N, S = future_NS.result()
        F = future_F.result()

    wS = (S - 1.75) * 0.25 * np.log10(N + 10)
    wFR = 0
    if N != 0:
        wFR = (2.18 / (N ** 0.4)) * ((0.4 * F) + (0.6 * R))
    PA = N * (1 + np.sqrt((wS ** 2) + (wFR ** 2)))

    return PA


def process_audio(input_path):
    try:
        v_pa = []
        fr, xr = wavfile.read(input_path)
        xr = xr / FIX_FACTOR
        xs = xr
        gcd = np.gcd(fr, new_sample_rate)
        if fr != sample_rate:
            up = new_sample_rate // gcd
            down = fr // gcd
            xs = resample_poly(xr, up, down)
        xr = xs
        chunk_length_samples = chunk_length_sec * sample_rate
        num_chunks = len(xr) // chunk_length_samples
    except Exception as e:
        print(f"Error: {e}")
        return []
    print("Processing audio...")
    try:
        for i in tqdm(range(num_chunks)):
            start_sample = i * chunk_length_samples
            end_sample = start_sample + chunk_length_samples
            chunk = xr[start_sample:end_sample]
            pa = PAAnalyzer(chunk)
            v_pa.append(pa)
        print("Audio processed.")
        return v_pa
    except Exception as e:
        print(f"Error: {e}")
        return []
   

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', type=str, default='data/lowPAsound.wav')
    argparser.add_argument('--output_path', type=str, default='data')
    argparser.add_argument('--output_filename', type=str, default='pas.json')
    args = argparser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    output_filename = args.output_filename
    if '.json' not in output_filename:
        output_filename += '.json'
    dataset_pas = {}

    v_pa = process_audio(input_path)
    dataset_pas[input_path.split('/')[-1]] = v_pa
    json.dump(dataset_pas, open(os.path.join(output_path, output_filename), 'w'), indent=4)