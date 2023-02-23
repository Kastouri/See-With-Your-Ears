# See With Your Ear

This project implements a binaural (3D) sound synthesis engine that allows a visually impaired person to "see" 
their environment by using sound localization.

A [more thorough documentation](https://kastouri.github.io/see_with_ear.html) (WIP) can be found in my website. 

# Try it!
All you need to try this tool is a stereo enabled headset and a laptop camera.
You can alternatively provide the path to a video or a URL of an ESP32-Cam webserver.

## How to install
You can clone this repository and set up a conda environment using the following commands:
```
git clone --recurse-submodules https://github.com/Kastouri/See-With-Your-Ears.git  # clone recursively
cd see_with_your_ears
git submodule update --init --recursive
conda env create -f environment.yml
```

## How to run
In order to run the alternative vision engine using you laptop camera use the following command.
Make sure you use a headset with stereo audio enable in order to get a feeling for 
the 3D sound effect.
```
python run_alternative_vision.py --video-source=0
```

## Notes:
- The synthesised sound doesn't reflect the correct field of view of the camera of most laptops. A wider field of view 
was used for the sake of this demo.
