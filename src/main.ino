// This program uses several functionalities and modifications 
// from the EdgeImpulse inferencing library.

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0


/* Includes ---------------------------------------------------------------- */
#include <PDM.h>
#include <training_kws_inference.h>
#include "neural_network.h"

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t buffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT];
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal

static uint8_t virtual_button = 0;
// const uint8_t button_1 = 2;
// const uint8_t button_2 = 3;
// const uint8_t button_3 = 4;
// const uint8_t button_4 = 5;
//uint8_t num_button = 0; // 0 represents none
bool button_pressed = false;

// Defaults: 0.3, 0.9
static NeuralNetwork myNetwork;
const float threshold = 0.6;

uint16_t num_epochs = 0;


/**
 * @brief      Arduino setup function
 */
void setup() {
    Serial.begin(9600);

    // Start button_configuration
    
    // pinMode(button_1, INPUT);
    // pinMode(button_2, INPUT);
    // pinMode(button_3, INPUT);
    // pinMode(button_4, INPUT);
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);

    digitalWrite(LED_BUILTIN, HIGH);

    if (microphone_setup(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
        ei_printf("ERR: Failed to setup audio sampling\r\n");
        return;
    }

    init_network_model();
    digitalWrite(LED_BUILTIN, LOW);

    num_epochs = 0;
}


void init_network_model() {
    char startChar;
    do {
        startChar = Serial.read();
        Serial.println("Waiting for new model...");
        delay(1000);
    } while(startChar != 's'); // s -> START

    delay(300);

    Serial.println("start");

    float learningRate = readFloat();
    float momentum = readFloat();
    
    Serial.write('n');

    myNetwork.initialize(learningRate, momentum);
    myNetwork.initWeights();

    char* myHiddenWeights = (char*) myNetwork.get_HiddenWeights();
    for (uint16_t i = 0; i < NFilter * 9; ++i) {
        Serial.write('n');
        while(Serial.available() < 4) {}
        for (int n = 0; n < 4; n++) {
            myHiddenWeights[i*4+n] = Serial.read();
        }
    }
    Serial.println("Received new client-side model.");
}

float readFloat() {
    byte res[4];
    while(Serial.available() < 4) {}
    for (int n = 0; n < 4; n++) {
        res[n] = Serial.read();
    }
    return *(float *)&res;
}

uint8_t readInt() {
    byte res[1];
    while(Serial.available() < 1) {}
    for (int n = 0; n < 1; n++) {
        res[n] = Serial.read();
    }
    return *(uint8_t *)&res;
}


void train(int nb, bool only_forward) {
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei::matrix_t features_matrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);

    // for(int i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
    //     Serial.print(inference.buffer[i]);
    //     Serial.print(",");
    // }
    // Serial.println();

    EI_IMPULSE_ERROR r = get_one_second_features(&signal, &features_matrix, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to get features (%d)\n", r);
        return;
    }
    // Sending inputs to Server 
    // float* myinput = features_matrix.buffer;
    // for (size_t i = 0; i < 50; i++) {
    //     for (size_t j = 0; j < 13; j++) {
    //         ei_printf_float(myinput[i*13 + j]);
    //         Serial.print(" ");
    //     }
    //     Serial.print("\r\n");
    // }
    

    myNetwork.forward(features_matrix.buffer);

    // Sending activation/label to Server 
    float* myOutput = myNetwork.get_output();
    for (size_t i = 0; i < HiddenNodes; i++) {
        ei_printf_float(myOutput[i]);
        Serial.print(" ");
    }
    Serial.print("\r\n");

    

    if (!only_forward) {

        // Sending label to Server 
        Serial.println(nb, DEC);

        // Receive error from server
        char* Error = (char*) myNetwork.get_Error();
        for (uint16_t i = 0; i < HiddenNodes; ++i) {
            Serial.write('n');
            while(Serial.available() < 4) {}
            for (int n = 0; n < 4; n++) {
                Error[i*4+n] = Serial.read();
            }
        }

        // BACKWARD
        myNetwork.backward(features_matrix.buffer);
        ++num_epochs;
    }

    // FORWARD
    // float forward_error = myNetwork.forward(features_matrix.buffer, myTarget);

    // float error = forward_error;
    // if (!only_forward) {
    //     error = backward_error;
    // }

    // float* myOutput = myNetwork.get_output();

    //uint8_t num_button_output = 0;
    //float max_output = 0.f;
    // Serial.print("Inference result: ");


    // Info to plot & graph!
    Serial.println("Done!");

    Serial.println(num_epochs, DEC);
    // Print outputs
    // for (size_t i = 0; i < 3; i++) {
    //     ei_printf_float(myOutput[i]);
    //     Serial.print(" ");
    //    if (myOutput[i] > max_output && myOutput[i] > threshold) {
    //        num_button_output = i + 1;
    //    }
    // }
    // Serial.print("\n");

    // Print error
    // ei_printf_float(error);
    // Serial.print("\n");

    // Serial.println(num_epochs, DEC);

    // char* myError = (char*) &error;
    // Serial.write(myError, sizeof(float));
    
    // Serial.println(nb, DEC);
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop() {
    digitalWrite(LEDR, HIGH);           // OFF
    digitalWrite(LEDG, HIGH);           // OFF
    digitalWrite(LEDB, HIGH);           // OFF
    digitalWrite(LED_BUILTIN, HIGH);    // ON

    // every loop wait for a set button.
    
    
    // Serial.write('n');

    digitalWrite(LEDR, LOW);           // OFF
    digitalWrite(LEDG, LOW);           // OFF
    digitalWrite(LEDB, LOW);           // OFF
    digitalWrite(LED_BUILTIN, LOW);    // ON

    uint8_t num_button = 0;

    bool only_forward = false;


    //get a signal from serial port to save as virtual_button


    if (virtual_button == 1 && (button_pressed == false || num_button == 1)) {
        // digitalWrite(LEDR, LOW);  //  ON
        Serial.println("bottom 1");
        num_button = 1;
        button_pressed = true;
        
        // while(virtual_button == 1) {}
        delay(200);
    }
    else if (virtual_button == 2 && (button_pressed == false || num_button == 2)) {
        // digitalWrite(LEDG, LOW);  //  ON
        Serial.println("bottom 2");
        num_button = 2;
        button_pressed = true;

        // while(virtual_button == 2) {}
        delay(200);
    }
    else if (virtual_button == 3 && (button_pressed == false || num_button == 3)) {
        // digitalWrite(LEDB, LOW);  //  ON
        Serial.println("bottom 3");
        num_button = 3;
        button_pressed = true;

        // while(virtual_button == 3) {}
        delay(200);
    }
    else if (virtual_button == 4 && (button_pressed == false || num_button == 4)) {
        digitalWrite(LEDR, LOW);  //  ON
        digitalWrite(LEDG, LOW);  //  ON
        digitalWrite(LEDB, LOW);  //  ON

        Serial.println("start_fl");
        // only_forward = true;

        // Debounce
        // while(virtual_button == 4 ) {}
        delay(200);
    } 
    
    // perform testing
    if (button_pressed == true) {
        Serial.println("Recording...");
        bool m = microphone_inference_record();
        if (!m) {
            Serial.println("ERR: Failed to record audio...");
            return;
        }
        Serial.println("Recording done");

        train(num_button, only_forward);

        button_pressed = false;

    } else {
        // while(Serial.available() < 1){} // We will not loop the read, but use blocking read 
        int read = Serial.read(); // loop reading
        if (read == 'a') { // s -> FEDERATED LEARNING
            /***********************
             * Federate Learning
             ***********************/
            
            Serial.write('y');
            // digitalWrite(LED_BUILTIN, HIGH);    // ON
            // delay(1000); // #TODO: Test whether this is more stable.
            while(Serial.available() < 1) {} //#TODO: Test whether this is more stable.
            if (Serial.read() == 's') {
                Serial.println("start");
                Serial.println(num_epochs, DEC);
                num_epochs = 0;

                /*******
                 * Sending model
                 *******/

                // Sending hidden layer
                // char* myHiddenWeights = (char*) myNetwork.get_HiddenWeights();
                char* myHiddenWeights = (char*) myNetwork.get_HiddenWeights();
                for (uint16_t i = 0; i < NFilter * 9; ++i) {
                    Serial.write(myHiddenWeights+i*sizeof(float), sizeof(float));
                }

                // Sending output layer
                // char* myOutputWeights = (char*) myNetwork.get_OutputWeights();
                // for (uint16_t i = 0; i < (HiddenNodes+1) * OutputNodes; ++i) {
                //     Serial.write(myOutputWeights+i*sizeof(float), sizeof(float));
                // }

                /*****
                 * Receiving model
                 *****/
                // Receiving hidden layer
                for (uint16_t i = 0; i < NFilter * 9; ++i) {
                    Serial.write('n');
                    while(Serial.available() < 4) {}
                    for (int n = 0; n < 4; n++) {
                        myHiddenWeights[i*4+n] = Serial.read();
                    }
                }

                // Receiving output layer
                // for (uint16_t i = 0; i < (HiddenNodes+1) * OutputNodes; ++i) {
                //     Serial.write('n');
                //     while(Serial.available() < 4) {}
                //     for (int n = 0; n < 4; n++) {
                //         myOutputWeights[i*4+n] = Serial.read();
                //     }
                // }
                // Serial.println("Model aggregation done");
            }

            // digitalWrite(LED_BUILTIN, LOW);    // OFF
            
        } else if (read == 't') { // Train with a sample
            Serial.println("ok");

            while(Serial.available() < 1) {}
            uint8_t label_send = Serial.read();
            Serial.print("Label Received "); 
            Serial.println(label_send);

            while(Serial.available() < 1) {}
            bool only_forward = Serial.read() == 1;
            // Serial.print("Only forward "); 
            Serial.println(only_forward);
            
            byte ref[2];
            for(int i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
                while(Serial.available() < 2) {}
                Serial.readBytes(ref, 2);
                inference.buffer[i] = 0;
                inference.buffer[i] = (ref[1] << 8) | ref[0];
            }
            Serial.print("Sample received for label ");
            Serial.println(label_send);
            train(label_send, only_forward);
            // Serial.println("Perform single sample training");
        } else if (read == 'b'){ // Use a signal to set button
            Serial.println("bt_set");
            virtual_button = readInt();
        }


    }
}


void ei_printf(const char *format, ...) {
    static char print_buf[1024] = { 0 };

    va_list args;
    va_start(args, format);
    int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
    va_end(args);

    if (r > 0) {
        Serial.write(print_buf);
    }
}


static void pdm_data_ready_inference_callback(void) {
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (inference.buf_ready == 0) {
        for(int i = 0; i < bytesRead>>1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];

            if(inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
                break;
            }
        }
    }
}


static bool microphone_setup(uint32_t n_samples) {
    inference.buf_count  = 0;
    inference.n_samples  = n_samples;
    inference.buf_ready  = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    // optionally set the gain, defaults to 20
    PDM.setGain(80);
    PDM.setBufferSize(4096);

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
        PDM.end();
        return false;
    }
    return true;
}


static bool microphone_inference_record(void) {
    inference.buf_ready = 0;
    inference.buf_count = 0;
    while(inference.buf_ready == 0) {
        delay(10);
    }
    return true;
}


static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
    return 0;
}
