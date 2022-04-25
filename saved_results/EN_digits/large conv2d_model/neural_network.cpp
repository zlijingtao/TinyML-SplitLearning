// This code is a modification of the code from http://robotics.hobbizine.com/arduinoann.html


#include <Arduino.h>
#include "neural_network.h"
#include <math.h>

void NeuralNetwork::initialize(float LearningRate, float Momentum) {
    this->LearningRate = LearningRate;
    this->Momentum = Momentum;
}

void NeuralNetwork::initWeights() {
    for (int oc = 0; oc < NFilter; oc++){
        for (int h = 0; h < 3; h++){
            for (int w = 0; w < 3; w++){
                ChangeHiddenWeights[oc*9  + h * 3 + w] = 0.0;
            }
        }
    }
}


void NeuralNetwork::forward(const float Input[]){
    /******************************************************************
    * Compute hidden layer activations and calculate activations
    ******************************************************************/
    for (int h = 0; h < n_H; h++){
        for(int w = 0 ; w < n_W ; w++ ) {  
            
            for (int oc = 0; oc < NFilter; oc++){
                
                float partial_sum = 0.0;

                partial_sum += HiddenWeights[oc*9 + 0] * Input[(h * Stride) * n_W_prev + w*Stride ];// h = 0, w = 0
                partial_sum += HiddenWeights[oc*9 + 1] * Input[(h * Stride) * n_W_prev + w*Stride + 1];// h = 0, w = 1
                partial_sum += HiddenWeights[oc*9 + 2] * Input[(h * Stride) * n_W_prev + w*Stride + 2];// h = 0, w = 2

                partial_sum += HiddenWeights[oc*9 + 3] * Input[(h * Stride + 1) * n_W_prev + w*Stride ];// h = 1, w = 0
                partial_sum += HiddenWeights[oc*9 + 4] * Input[(h * Stride + 1) * n_W_prev + w*Stride + 1];// h = 1, w = 1
                partial_sum += HiddenWeights[oc*9 + 5] * Input[(h * Stride + 1) * n_W_prev + w*Stride + 2];// h = 1, w = 2

                partial_sum += HiddenWeights[oc*9 + 6] * Input[(h * Stride + 2) * n_W_prev + w*Stride ];// h = 2, w = 0
                partial_sum += HiddenWeights[oc*9 + 7] * Input[(h * Stride + 2) * n_W_prev + w*Stride + 1];// h = 2, w = 1
                partial_sum += HiddenWeights[oc*9 + 8] * Input[(h * Stride + 2) * n_W_prev + w*Stride + 2];// h = 2, w = 2

                // partial_sum += bias
                if (partial_sum <= 0){
                    Hidden[h * n_W * NFilter + w * NFilter + oc] = 0.0;
                }
                else{
                    Hidden[h * n_W * NFilter + w * NFilter + oc] = partial_sum;
                }
            }
        }
    }
}

void NeuralNetwork::backward(const float Input[]){
    /******************************************************************
    * Backpropagate errors to hidden layer
    ******************************************************************/


    for(int i = 0 ; i < HiddenNodes ; i++ ) {
        // RELU
        if (Hidden[i] <= 0){
            HiddenDelta[i] = 0.0;
        }
        else{
            HiddenDelta[i] = Error[i];
        }
        
    }
    
    //Apply Momentum update base
    for (int oc = 0; oc < NFilter; oc++){
        for (int f = 0; f < 9; f++){
                ChangeHiddenWeights[oc*9  + f] = Momentum * ChangeHiddenWeights[oc*9  + f];
        }
    }

    //Calculate Updates
    for (int h = 0; h < n_H; h++){
        for(int w = 0 ; w < n_W ; w++ ) {  
            
            for (int oc = 0; oc < NFilter; oc++){
                


                ChangeHiddenWeights[oc * 9 + 0] += (1 - Momentum)* LearningRate * Input[(h * Stride) * n_W_prev + w*Stride ] * HiddenDelta[h*n_W*NFilter + w*NFilter+oc];
                ChangeHiddenWeights[oc * 9 + 1] += (1 - Momentum)* LearningRate * Input[(h * Stride) * n_W_prev + w*Stride +1] * HiddenDelta[h*n_W*NFilter + w*NFilter+oc];
                ChangeHiddenWeights[oc * 9 + 2] += (1 - Momentum)* LearningRate * Input[(h * Stride) * n_W_prev + w*Stride +2] * HiddenDelta[h*n_W*NFilter + w*NFilter+oc];

                ChangeHiddenWeights[oc * 9 + 3] += (1 - Momentum)* LearningRate * Input[(h * Stride+1) * n_W_prev + w*Stride ] * HiddenDelta[h*n_W*NFilter + w*NFilter+oc];
                ChangeHiddenWeights[oc * 9 + 4] += (1 - Momentum)* LearningRate * Input[(h * Stride+1) * n_W_prev + w*Stride +1] * HiddenDelta[h*n_W*NFilter + w*NFilter+oc];
                ChangeHiddenWeights[oc * 9 + 5] += (1 - Momentum)* LearningRate * Input[(h * Stride+1) * n_W_prev + w*Stride +2] * HiddenDelta[h*n_W*NFilter + w*NFilter+oc];

                ChangeHiddenWeights[oc * 9 + 6] += (1 - Momentum)* LearningRate * Input[(h * Stride+2) * n_W_prev + w*Stride ] * HiddenDelta[h*n_W*NFilter + w*NFilter+oc];
                ChangeHiddenWeights[oc * 9 + 7] += (1 - Momentum)* LearningRate * Input[(h * Stride+2) * n_W_prev + w*Stride +1] * HiddenDelta[h*n_W*NFilter + w*NFilter+oc];
                ChangeHiddenWeights[oc * 9 + 8] += (1 - Momentum)* LearningRate * Input[(h * Stride+2) * n_W_prev + w*Stride +2] * HiddenDelta[h*n_W*NFilter + w*NFilter+oc];

            }
        }
    }

    //Apply Updates
    for (int oc = 0; oc < NFilter; oc++){
        for (int f = 0; f < 9; f++){
                HiddenWeights[oc*9  + f] -= ChangeHiddenWeights[oc*9  + f];
        }
    }
}


float* NeuralNetwork::get_output(){
    return Hidden;
}

float* NeuralNetwork::get_HiddenWeights(){
    return HiddenWeights;
}

// float* NeuralNetwork::get_HiddenWeights(){
//     return HiddenWeights;
// }

float* NeuralNetwork::get_Error(){
    return Error;
}