#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <time.h>


// 2 weight + 1 bias
#define WEIGHTS_STRUCT_SIZE 3
typedef float weights_strct[WEIGHTS_STRUCT_SIZE];
// 2 inputs + 1 output
#define TRAIN_STRUCT_SIZE 3
typedef float train_struct[TRAIN_STRUCT_SIZE];


#define WEIGHT_SIZE 3
#define TRAIN_DATA_SIZE 4

train_struct or_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

train_struct and_train[] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1},
};

train_struct xor_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

train_struct nand_train[] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

train_struct *train_data = or_train;

float randf(void)
{
    return (float) rand()/ (float) RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

// calculate neuron
float calculate_neurons(weights_strct weight, float inputs[], size_t size){
    float output = 0.0f;
    for(int i=0; i < size; i++){
        output+=weight[i]*inputs[i];
    }
    output+=weight[size];
    return sigmoidf(output);
}

// forward function 
float forward(weights_strct weights[], float inputs[], size_t size){
    float n1 = calculate_neurons(weights[0], inputs,size);
    float n2 = calculate_neurons(weights[1], inputs, size);
    float n3input[] = {n1, n2};
    return calculate_neurons(weights[2], n3input, size);
}

// cost function
float cost(weights_strct weights[], train_struct inputs[]){
    float cost = 0.0f;

    for(size_t i = 0; i < TRAIN_DATA_SIZE; i++){
        float expected = inputs[i][TRAIN_STRUCT_SIZE-1];
        float predicted = forward( weights, inputs[i],2);
        float diff = predicted - expected;
        cost+=diff*diff;
    }

    cost /= TRAIN_DATA_SIZE;

    return cost;
}


// initializing weight + 1 biase 
void init_weights(weights_strct* weight, size_t size){
    for(size_t i=0; i < size; i++){
        for(size_t j=0; j < WEIGHTS_STRUCT_SIZE; j++){
            weight[i][j] = randf();
        }
    }
}

void print_weight(weights_strct weight[], size_t size){
    for(size_t i=0; i < size; i++){
        printf("n%zu - ,",i);
        for(size_t j=0; j < (WEIGHTS_STRUCT_SIZE)-1; j++){
            printf("w[%zu] = %f , ",j,weight[i][j]);
        }
        printf("n[%zu]b %f\n", i, weight[i][WEIGHTS_STRUCT_SIZE-1]);
    }
}


// Find the diffrence and send the data
void find_diff(weights_strct *diff,weights_strct weights[], train_struct inputs[],float eps){
    float c = cost(weights, inputs);
    for(size_t i = 0; i < WEIGHT_SIZE; i++){
        for(size_t j = 0; j < WEIGHTS_STRUCT_SIZE; j++){
            float original = weights[i][j];
            weights[i][j] += eps;
            diff[i][j] = ( cost(weights,inputs) - c )/eps;
            weights[i][j] = original;
        }
    }
    return;
}

// For Learning
void learn(int itrations, weights_strct *weights, train_struct inputs[], float learning_rate, float eps){
    for(int i = 0; i < itrations; i++){
        float c = cost(weights, inputs);
        // printf("c=%f\n", c);
        weights_strct *diff = malloc(sizeof(weights_strct)*WEIGHT_SIZE);
        find_diff(diff, weights, inputs,eps);
        for ( int i = 0; i < WEIGHT_SIZE; i++){
            for( int j = 0; j < WEIGHTS_STRUCT_SIZE; j++){
                weights[i][j]-=learning_rate*diff[i][j];
            }
        }
        
        free(diff);
    }   
}

int main(){
    srand(time(0));
    weights_strct w[WEIGHT_SIZE];
    init_weights(w,WEIGHT_SIZE);
    printf("Initial Weights of 3 neurons with bias\n");
    print_weight(w, WEIGHT_SIZE);

    printf("\ninitial cost = %f\n", cost(w,train_data));
    learn(10000*100, w, train_data, 1e-1f, 1e-4);
    printf("-----------FINAL WEIGHT------------------\n");
    print_weight(w,WEIGHT_SIZE);
    printf("\nFinal cost = %f\n", cost(w,train_data));
    printf("\n");
    for(int i=0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            float input[] = {(float)i, (float)j};
            float output = forward(w,input,2);
            printf("%d | %d = %f \n", i, j, output);
        }
    }
    return 0;
}