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
            float Rando = float(random(100))/100;
            HiddenWeights[j*HiddenNodes + i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
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
            // Accum += Input[j] * HiddenWeights[j*HiddenNodes + i]; //Original
            Accum += Input[j]/100. * HiddenWeights[j*HiddenNodes + i]; // Input scale down by 100
        }
        // Changed the activation to RELU
        // if (Accum >= 0){
        //     Hidden[i] = Accum;
        // }
        // else {
        //     Hidden[i] = 0.0;
        // }
        // Original sigmoid activation
        // Hidden[i] = 1.0 / (1.0 + exp(-Accum));

        // Changed the activation to identity
        Hidden[i] = Accum;
    }
}

void NeuralNetwork::backward(const float Input[]){

    /******************************************************************
    * Backpropagate errors to hidden layer
    ******************************************************************/
    for(int i = 0 ; i < HiddenNodes ; i++ ) {
        // Changed the dW calculation to RELU (backward)   
        // if (Error[i] < 0){
        //     HiddenDelta[i] = 0.0;
        // }
        // else{
        //     HiddenDelta[i] = Error[i] * Hidden[i];
        // }
        // Original sigmoid activation (backward)
        // HiddenDelta[i] = Error[i] * Hidden[i] * (1.0 - Hidden[i]) ;

        // Changed the dW calculation to Identity (backward)   
        HiddenDelta[i] = Error[i] * Hidden[i];
    }

    /******************************************************************
    * Update Inner-->Hidden Weights
    ******************************************************************/
    for(int i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes*HiddenNodes + i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        HiddenWeights[InputNodes*HiddenNodes + i] += ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        for(int j = 0 ; j < InputNodes ; j++ ) { 
            // ChangeHiddenWeights[j*HiddenNodes + i] = LearningRate * Input[j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j*HiddenNodes + i];   // Original
            ChangeHiddenWeights[j*HiddenNodes + i] = LearningRate * Input[j]/100. * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j*HiddenNodes + i];  // Input scale down by 100
            HiddenWeights[j*HiddenNodes + i] += ChangeHiddenWeights[j*HiddenNodes + i] ;
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