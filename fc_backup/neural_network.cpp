// This code is a modification of the code from http://robotics.hobbizine.com/arduinoann.html


#include <Arduino.h>
#include "neural_network.h"
#include <math.h>

void NeuralNetwork::initialize(float LearningRate, float Momentum) {
    this->LearningRate = LearningRate;
    this->Momentum = Momentum;
}

void NeuralNetwork::initWeights() {
    for(int i = 0 ; i < HiddenNodes ; i++ ) {    
        for(int j = 0 ; j <= InputNodes ; j++ ) { 
            ChangeHiddenWeights[j*HiddenNodes + i] = 0.0 ;
        }
    }
}


void NeuralNetwork::forward(const float Input[]){
    /******************************************************************
    * Compute hidden layer activations and calculate activations
    ******************************************************************/
    for (int i = 0; i < HiddenNodes; i++) {
        float Accum = HiddenWeights[InputNodes*HiddenNodes + i];
        for (int j = 0; j < InputNodes; j++) {
            Accum += Input[j] * HiddenWeights[j*HiddenNodes + i]; 
        }
        if (Accum <= 0){
            Hidden[i] = 0.0;
        }
        else{
            Hidden[i] = Accum;
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
        // identity
        // HiddenDelta[i] = Error[i];
        
    }
    /******************************************************************
    * Update Inner-->Hidden Weights
    ******************************************************************/
    for(int i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes*HiddenNodes + i] = (1 - Momentum) * LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        HiddenWeights[InputNodes*HiddenNodes + i] -= ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        for(int j = 0 ; j < InputNodes ; j++ ) { 
            ChangeHiddenWeights[j*HiddenNodes + i] = (1 - Momentum) * LearningRate * Input[j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j*HiddenNodes + i];   // Original
            HiddenWeights[j*HiddenNodes + i] -= ChangeHiddenWeights[j*HiddenNodes + i] ;
        }
    }
}


float* NeuralNetwork::get_output(){
    return Hidden;
}

float* NeuralNetwork::get_HiddenWeights(){
    return HiddenWeights;
}

float* NeuralNetwork::get_Error(){
    return Error;
}