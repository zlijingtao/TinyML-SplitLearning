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
            Accum += Input[j] * HiddenWeights[j*HiddenNodes + i];
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum));
    }
}

void NeuralNetwork::backward(const float Input[], const float Error[]){

    /******************************************************************
    * Backpropagate errors to hidden layer
    ******************************************************************/
    for(int i = 0 ; i < HiddenNodes ; i++ ) {    
        HiddenDelta[i] = Error[i] * Hidden[i] * (1.0 - Hidden[i]) ;
    }

    /******************************************************************
    * Update Inner-->Hidden Weights
    ******************************************************************/
    for(int i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes*HiddenNodes + i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        HiddenWeights[InputNodes*HiddenNodes + i] += ChangeHiddenWeights[InputNodes*HiddenNodes + i] ;
        for(int j = 0 ; j < InputNodes ; j++ ) { 
            ChangeHiddenWeights[j*HiddenNodes + i] = LearningRate * Input[j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j*HiddenNodes + i];
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