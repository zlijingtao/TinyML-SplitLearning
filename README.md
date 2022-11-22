# Split Federated Learning on Micro-controllers

Final project for the CEN598 course of Arizona State University. We showcase the strength of SFL in Arduino Nano 33 BLE Sense.

Report is arxived here: [Split Federated Learning on Micro-controllers: A Keyword Spotting Showcase](https://arxiv.org/abs/2210.01961)

## TODO:
1. Extend the server-side model to more complex architecture
2. Extend the serial communication via bluetooth


## Python Package

'''
pip install pyserial numpy matplotlib
'''

## How tu use it
1. Open the project with PlatformIO and flash the firmware to all the boards
2. Run the sfl_server.py using Python3 (This is a simulated training process). The default port should be /dev/ttyACM0 or /dev/ttyACM1

## Run Federated Learning
1. Checkout "fl" branch of this repo
2. Run the fl_server.py using Python3 (This is a simulated training process). The default port should be /dev/ttyACM0 or /dev/ttyACM1

## Authors
- Jingtao Li
- Runcong Kuang

## Acknowledgement

This project is originated from Federated Learning with Arduino Nano 33 BLE Sense, which is authored by:
- Marc Monfort
- Nil Llisterri
