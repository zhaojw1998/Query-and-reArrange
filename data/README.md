## Dataset 

We release our processed datasets, as well as scripts for data processing, to facilitate follow-up research.

We use a total of 4 datasets in our work: 
* piano arrangement [POP909](https://github.com/music-x-lab/POP909-Dataset);
* multi-track arrangement [Slakh2100](https://zenodo.org/record/4599666);
* Bach Chorales in [Music21](https://pypi.org/project/music21/);
* String Quartets in [MusicNet](https://zenodo.org/record/5120004).

### File Directory
Our data folder is organized as follows

```
./data/
    ├──POP909/                      processed POP909 dataset
        ├──train/
            ├──001.npz
            └──...
        ├──validation/
            ├──005.npz
            └──...
        └──test/   
            ├──003.npz
            └──...
    ├──Slakh2100/                   processed Slakh2100 dataset
        ├──train/
            ├──Track00001.npz
            └──...
        ├──validation/
            ├──Track01501.npz
            └──...
        └──test/   
            ├──Track01876.npz
            └──...
    ├──Bach_Chorales/               processed Bach Chorale dataset
        ├──bwv1.6.npz 
        └──...  
    ├──String_Quartets/             processed String Quartets dataset
        ├──Beethoven-2313_qt15_1.npz 
        └──...  
    │    
    ├──fold_record_chorales.txt     10-fold split record for Bach Chorales 
    │   
    ├──fold_record_quartets.txt     10-fold split record for String Quartets
    │ 
    ├──bach_chorales.py             data processing scripts for Bach Chorales
    │   
    ├──pop909.py                    data processing scripts for POP909
    │   
    ├──slakh2100.py                 data processing scripts for Slakh2100
    │   
    └──string_quartets.py           data processing scripts for String Quartets
```