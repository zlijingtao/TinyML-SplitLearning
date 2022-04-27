#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK


/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

static const int InputNodes = 650;
static const int n_H_prev = 13;
static const int n_W_prev = 50;
static const int NFilter = 12;
static const int Stride = 2;
static const int n_H = (n_H_prev - 3) / Stride + 1;
static const int n_W = (n_W_prev - 3) / Stride + 1;
static const int HiddenNodes = NFilter * n_H * n_W;
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
        // float Hidden[]
        float HiddenWeights[NFilter*9] = {};
        float Error[HiddenNodes] = {};
        float HiddenDelta[HiddenNodes] = {};
        float ChangeHiddenWeights[NFilter*9] = {};

        float LearningRate = 0.01;
        float Momentum = 0.0;

        
};


#endif
