#include <atomic>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include <fcntl.h>
#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>


using namespace std;

// Define Stack class
template <typename T>
class Stack
{
private:
    vector<T> elements;

public:
    void push(T const &element) { elements.push_back(element); }
    void pop() { elements.pop_back(); }
    T peek() const { return elements.back(); }
    bool isEmpty() const { return elements.empty(); }
    void display() const
    {
        for (const auto &element : elements)
        {
            cout << element << " ";
        }
        cout << endl;
    }
};

#define N_LAYERS 5

sem_t lock;
Stack<double> st; // stack for tracking values

double Output;

double valuesArr[2]; // array of x1 and x2 (formulas)

double outputLayer[1][8];
double inputLayerWeights[2][1][8];
double layersAns[1][8];
double layerWeights[5][8][8];
double inputValues[2];


void initializeWeights()
{
    // Seed for random number generation
    srand(time(0));

    // Initialize inputLayerWeights
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 1; ++j)
        {
            for (int k = 0; k < 8; ++k)
            {
                inputLayerWeights[i][j][k] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }
        }
    }

    // Initialize layerWeights
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            for (int k = 0; k < 8; ++k)
            {
                layerWeights[i][j][k] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }
        }
    }

    // Initialize outputLayer
    for (int i = 0; i < 1; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            outputLayer[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }

    // Initialize layersAns with zeros
    for (int i = 0; i < 1; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            layersAns[i][j] = 0.0;
        }
    }

    // Initialize inputValues
    inputValues[0] = 0.1;
    inputValues[1] = 0.2;
}

struct Neuron
{
    int NeuronNum;
    int weights;
};

void critical_Section(int id)
{
    const int num_rows = 1;
    const int num_cols = 8;

    int fd = open("my_pipe", O_RDONLY); // Open pipe for reading
    sem_wait(&lock);

    // Read the values from the pipe
    for (int i = 0; i < num_rows; ++i)
    {
        read(fd, layersAns[i], sizeof(layersAns[i]));
    }
    close(fd);

    double val = 0.0;
    double dummy[num_rows][num_cols];

    // Perform matrix multiplication
    for (int i = 0; i < num_cols; ++i) // Iterate over layer rows
    {
        for (int k = 0; k < num_cols; ++k) // Iterate over ans values
        {
            for (int j = 0; j < num_cols; ++j) // Iterate over layer columns
            {
                val += layersAns[0][k] * layerWeights[id][i][j];
            }
        }
        dummy[0][i] = val;
        val = 0;
    }

    // Push the results onto the stack
    for (int i = 0; i < num_cols; ++i)
    {
        layersAns[0][i] = dummy[0][i];
        st.push(layersAns[0][i]);
    }

    // Open pipe for writing and send the updated values
    fd = open("my_pipe", O_WRONLY);
    for (int i = 0; i < num_rows; ++i)
    {
        write(fd, layersAns[i], sizeof(layersAns[i]));
    }
    close(fd);

    sem_post(&lock); // Release the semaphore
}


struct network_Layers
{
    int num_layers; // total number of layers
private:
   // int num_layers;
    double layersAns[1][8];
    double inputLayerWeights[2][1][8];
    double inputValues[2];
    Stack<double> st;  // Assuming you have a Stack class defined

   void performCriticalSection(int id)
{
    int fd = open("my_pipe", O_RDONLY); // creating pipe for reading and writing
    sem_wait(&lock);
    for (int i = 0; i < 1; i++) // rows of answer array [1][8 ]
    {
        read(fd, layersAns[i], sizeof(layersAns[i]));
    }
    close(fd);

    double val = 0.0;
    double dummy[1][8];
    for (int i = 0; i < 8; i++) // layer rows
    {
        for (int k = 0; k < 8; k++) // ans values
        {
            for (int j = 0; j < 8; j++) // layer col
            {
                val += layersAns[0][k] * layerWeights[id][i][j];
            }
        }
        dummy[0][i] = val;
        val = 0;
    }
    for (int i = 0; i < 8; i++)
    {
        layersAns[0][i] = dummy[0][i];
        st.push(layersAns[0][i]);
    }

    fd = open("my_pipe", O_WRONLY);
    for (int i = 0; i < 1; i++)
    {
        write(fd, layersAns[i], sizeof(layersAns[i]));
    }
    close(fd);
    sem_post(&lock);
}

     void writeToPipe()
    {
        int fd = open("my_pipe", O_WRONLY); // for writing only

        for (int i = 0; i < 1; i++)
        {
            write(fd, layersAns[i], sizeof(layersAns[i]));
        }

        close(fd);
    }
        double extractSignificand(double x)
    {
        std::stringstream ss;
        ss << x;
        std::string str = ss.str();
        return std::stod(str.substr(0, str.find('e')));
    }

    void pushInputLayerToStack()
    {
        for (int i = 0; i < 8; i++)
        {
            st.push(layersAns[0][i]);
        }
    }

    
    void generateInputLayer()
    {
        for (int k = 0; k < 2; k++)
        {
            for (int j = 0; j < 8; j++)
            {
                for (int i = 0; i < 2; i++)
                {
                    layersAns[0][j] += inputValues[i] * inputLayerWeights[k][0][j];
                }
            }
        }
    }


public:
    network_Layers(int num) : num_layers(num)
    {
        generateInputLayer();
        pushInputLayerToStack();
        writeToPipe();
    }

    static void* performCriticalSectionWrapper(void* arg) {
        // Cast arg and call the member function
        std::pair<network_Layers*, int>* pair = (std::pair<network_Layers*, int>*)arg;
        network_Layers* self = pair->first;
        int id = pair->second;
        self->performCriticalSection(id);
        delete pair; // Clean up the heap memory
        return nullptr;
    }

    double for1(double x)
    {
        double value = ((x * x) + x + 1) / 2;
        return extractSignificand(value);
    }

    double for2(double x)
    {
        double value = ((x * x) - x) / 2;
        return extractSignificand(value);
    }

 void forwardPropagation() {
        std::cout << "Forward Propagation in process\n";
        sleep(2); // POSIX sleep for 2 seconds

        // Array to hold thread IDs
        pthread_t processes[N_LAYERS];

        // Start child threads in a loop
        for (int i = 0; i < num_layers; i++) {
            // We need to pass 'this' pointer and the loop index 'i' to the pthread function
            std::pair<network_Layers*, int>* pair = new std::pair<network_Layers*, int>(this, i);
            if (pthread_create(&processes[i], NULL, &network_Layers::performCriticalSectionWrapper, (void*)pair) != 0) {
                std::cerr << "Error in creating thread" << std::endl;
                exit(1);
            }
        }

        // Wait for all child threads to complete
        for (int i = 0; i < num_layers; i++) {
            pthread_join(processes[i], NULL);
        }
                // Output layer multiplication
        for (int j = 0; j < 8; j++)
        {
            for (int i = 0; i < 8; i++)
            {
                layersAns[0][j] += layersAns[0][i] * outputLayer[0][j];
            }
            st.push(layersAns[0][j]);
        }
    }
void backPropagation()
{
    // Calculate the final output from the answer array
    double finalOutput = 0.0;
    for (int i = 0; i < 8; i++)
    {
        finalOutput += layersAns[0][i];
    }

    // Print the final output after forward propagation
    std::cout << "\n Calculating output post forward propagation: " << finalOutput << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // Start back propagation processing
    std::cout << "Starting Back Propagation " << std::endl;
    sleep(2);

    // Transform and write values to the pipe
    valuesArr[0] = for1(finalOutput);
    valuesArr[1] = for2(finalOutput);
    int fd = open("values", O_WRONLY);
    write(fd, valuesArr, sizeof(valuesArr));
    close(fd);

    // Print the transformed values on output
    std::cout << "Changed output values : ";
    std::cout << "x1 = " << valuesArr[0] << ", ";
    std::cout << "x2 = " << valuesArr[1] << std::endl;

    // Process critical section for each layer
    int totalLayers = num_layers + 1 + 1; // input layer + remaining layers + output layer
    for (int i = totalLayers; i > 0; i--)
    {
        std::cout << "---- Currently Processing  " << i << " Layer values----" << std::endl;
        backPropagationCriticalSection();
    }

    // Read the last values after multiplying with input values
    fd = open("values", O_RDONLY);
    read(fd, valuesArr, sizeof(valuesArr));
    close(fd);

    // Transform and print the last values on input
    valuesArr[0] = for1(valuesArr[0]);
    valuesArr[1] = for2(valuesArr[1]);
    std::cout << "\nx1 = " << valuesArr[0] << std::endl;
    std::cout << "x2 = " << valuesArr[1] << std::endl;

    // Update input values
    inputValues[0] = valuesArr[0];
    inputValues[1] = valuesArr[1];
}
void backPropagationCriticalSection()
{
    sem_wait(&lock);

    // Print values from the stack
    int counter = 0;
    std::cout << "stack values : ";
    while (counter != 8 && !st.isEmpty())
    {
        std::cout << st.peek() << " ";
        st.pop();
        counter++;
    }
    std::cout << std::endl;

    // Read values from the pipe
    int fd = open("values", O_RDONLY);
    read(fd, valuesArr, sizeof(valuesArr));
    close(fd);

    // Apply formulas to the values
    valuesArr[0] = for1(valuesArr[0]);
    valuesArr[1] = for2(valuesArr[1]);

    // Print the transformed values
    std::cout << "Changed values: ";
    std::cout << "x1 = " << valuesArr[0] << ", ";
    std::cout << "x2 = " << valuesArr[1] << std::endl;

    // Write the transformed values back to the pipe
    int fd2 = open("values", O_WRONLY);
    write(fd2, valuesArr, sizeof(valuesArr));
    close(fd2);

    sem_post(&lock);
}



    void printOutput()
    {
        st.display();
    }
void startProcessing()
{
    sem_t processLock;
    if (sem_init(&processLock, 0, 1) != 0)
    {
        cerr << "Error initializing semaphore";
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 1; i++)
    {
        pid_t id = fork();

        // Error handling for fork()
        if (id < 0)
        {
            cout<<"Error Process Creation"<<endl;
            exit(EXIT_FAILURE);
        }

        // Child process
        else if (id == 0)
        {
            sem_wait(&processLock);

            // Critical Section
            forwardPropagation();
            backPropagation();

            sem_post(&processLock);
            exit(EXIT_SUCCESS);
        }

        // Parent process
        else if (id > 0)
        {
            // Close semaphore in parent process to avoid resource leaks
            sem_close(&processLock);

            // Wait for the child process to complete
            waitpid(id, NULL, 0);
        }
    }

    // Destroy the semaphore after all processes are done
    sem_destroy(&processLock);
}

};

int main()
{
    // Initialize weights
initializeWeights();

    // Initialize semaphore
    if (sem_init(&lock, 0, 1) != 0)
    {
        cerr << "Error initializing semaphore" << endl;
        return EXIT_FAILURE;
    }

    // Create and process network layers
    network_Layers obj(N_LAYERS); // Number of layers
    obj.startProcessing();

    // Destroy the semaphore after use
    if (sem_destroy(&lock) != 0)
    {
        cerr << "Error destroying semaphore" << endl;
        return EXIT_FAILURE;
    }

    cout << "Program completed successfully." << endl;
 ///for(int)
    return EXIT_SUCCESS;
}
