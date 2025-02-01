# PYPAAnalyzer



## PYPAAnalyzer is a MOD Python version of a URPAAnalyzer

Matlab and original version: https://github.com/jausegar/urbauramon/tree/master/URPAA


## How to Use

You can extract psychoacoustic annoyance from audios with different durations using paanalyzer_folder.py --input_folder <folder> --output_path <output json path> --output_filename <filename of json output>

You can also extract psychoacoustic annoyance from single audio using paanalyzer_path.py --input_path <folder> --output_path <output json path> --output_filename <filename of json output>

## Requirements:
- python 3.12

- numpy==2.2.2

- scipy==1.15.1

- tqdm==4.67.1

You can use pip install -r requirements.txt to install dependecies

## Credits and License

If you use this Audio Analyzer for your publications please cite their SEA papers:


"Zwicker’s Annoyance model implementation in a WASN node" A. Pastor-Aparicio, J. Lopez-Ballester, J. Segura-Garcia, S. Felici-Castell, M. Cobos-Serrano, R. Fayos-Jordán, J.J. Pérez-Solano. INTERNOISE 2019, 48th International Congress and Exhibition on Noise Control Engineering

This is the source distribution MOD of URPAA: Urbauramon Psycho-acoustic Annoyance Analyzer licensed under the GPLv3+. Please consult the file COPYING for more information about this license.


## Authors

Gustavo "Joesty" Ribeiro
