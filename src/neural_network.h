#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK


/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

static const int InputNodes = 650;
static const int HiddenNodes = 25;
static const float InitialWeightMax = 0.5;

class NeuralNetwork {
    public:

        void initialize(float LearningRate, float Momentum);
        // ~NeuralNetwork();

        void initWeights();
        void forward(const float Input[]);
        void backward(const float Input[]);

        float* get_output();

        float* get_HiddenWeights();
        
        float* get_Error();
        
    private:

        float Hidden[HiddenNodes] = {};
        float HiddenWeights[(InputNodes+1) * HiddenNodes] = {};
        float Error[HiddenNodes] = {};
        float HiddenDelta[HiddenNodes] = {};
        float ChangeHiddenWeights[(InputNodes+1) * HiddenNodes] = {};

        float LearningRate = 0.01;
        float Momentum = 0.9;
};


#endif
