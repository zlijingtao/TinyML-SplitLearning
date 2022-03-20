# Split Federated Learning with Arduino Nano 33 BLE Sense

Final project for the CEN598 course of Arizona State University

## TODO:
1. Check the correctness of the code and the training process
2. Done testing "Real-case Usage" (refer to below)
3. Extend the server-side model to more complex architecture
4. Build our own dataset, taking experience from P4
5. Extend the serial communication via bluetooth


## Python Package

'''
pip install pyserial numpy matplotlib
'''

## How tu use it
1. Open the project with PlatformIO and flash the firmware to all the boards
2. Run the sfl_server.py using Python3 (This is a simulated training process). The default port should be /dev/ttyACM0 or /dev/ttyACM1

## Real-case Usage
3. Start training the devices using the virtual buttons (secified in sfl_server.py), however, below are untested
    * The 3 buttons on the left are used to train 3 different keywords (to be decided by you!)
    * The board will start recording when the button is pressed & RELEASED (one second)
    * The fourth button can be used to start the Federated Learning process or to only run the inference without training. It can be configured on the main loop in src/main.ino

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