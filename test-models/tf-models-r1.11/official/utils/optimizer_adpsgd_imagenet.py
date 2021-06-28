
import deep500 as d5
from deep500.frameworks import tensorflow as d5tf
import os
import tensorflow as tf

class DistSGDOptimizer(object):
    def __init__(self, optimizer: tf.train.Optimizer, comm_size: int):
        self.comm_size = comm_size
        self.optimizer = optimizer

        # Compile the operator
        opdesc = d5.compile_custom_cppop_inline('allreducef', _sallreduce,
                                                # Input tensor shapes (gradient, UNUSED last gradient op)
                                                [d5.tensordesc.runtime_shape(tf.float32), d5.tensordesc.runtime_shape(tf.float32)],
                                                # Output tensor shapes (reduced gradient)
                                                [d5.tensordesc.runtime_shape(tf.float32)],
                                                live_output=True, output_folder='/tmp')
        self.compiled_op = d5tf.custom_op(opdesc, compile_only=True)
        self._handles = []

    def compute_gradients(self, *args, **kwargs):
        return self.optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, global_step):

        optimizer =  self.optimizer
        #new_gvs = []
        last_var = None
        dependencies = []
        for grad, var in reversed(grads_and_vars):
            if grad is None:
                #new_gvs.append((grad, var))
                continue
            else:
                if last_var is None:
                    last_var = var
                    #last_var = grad
                    
            self.compiled_op.op.inputs = [d5tf.desc_from_tensor(var), d5tf.desc_from_tensor(last_var)]
            self.compiled_op.op.outputs = [d5tf.desc_from_tensor(var)]
            op, lib, handle = d5tf.custom_op(self.compiled_op, return_handle=True)
            self._handles.append((lib, handle))
            
            with tf.control_dependencies([grad]):
                #model_avg = op((var/self.comm_size), last_var)
                model_avg = op(var, last_var)

            assign_op = tf.assign(var, model_avg)
            dependencies.append(assign_op)
            #var.assign(model_avg)
            #with tf.control_dependencies([assign_op]):
            #    new_gvs.append((grad, var))
            last_var = model_avg

         
        with tf.control_dependencies([dep for dep in dependencies]):
            opt_ret = self.optimizer.apply_gradients(grads_and_vars, global_step)
        return opt_ret




_sallreduce = """
#include <deep500/deep500.h>
#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>


#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>


int counter = 0;
int comm_thread_counter = 0;
int comm_round_counter = 0;
pthread_t cthread;


//number of allreduces for one step of ResNet-50
#define OPS 161
int length[OPS] = {
1001,
2050048,
2048,
2048,
1048576,
512,
512,
2359296,
512,
512,
1048576,
2048,
2048,
1048576,
512,
512,
2359296,
512,
512,
1048576,
2048,
2048,
1048576,
512,
512,
2359296,
512,
512,
524288,
2048,
2048,
2097152,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
262144,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
262144,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
262144,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
262144,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
262144,
1024,
1024,
262144,
256,
256,
589824,
256,
256,
131072,
1024,
1024,
524288,
512,
512,
65536,
128,
128,
147456,
128,
128,
65536,
512,
512,
65536,
128,
128,
147456,
128,
128,
65536,
512,
512,
65536,
128,
128,
147456,
128,
128,
65536,
512,
512,
65536,
128,
128,
147456,
128,
128,
32768,
512,
512,
131072,
256,
256,
16384,
64,
64,
36864,
64,
64,
16384,
256,
256,
16384,
64,
64,
36864,
64,
64,
16384,
256,
256,
16384,
64,
64,
36864,
64,
64,
4096,
256,
256,
16384,
64,
64,
9408
};


//pointers of the input/output buffers
float* output_buff[OPS];
float* input_buff[OPS];
volatile float* shared_model_buff[OPS];

int max_int = 2147483624;
pthread_spinlock_t lock[OPS];

class allreducef : public deep500::CustomOperator {
  protected:
    int m_len;
    int64_t m_totalbytes;
  public:
    allreducef(int len) : m_len(len), m_totalbytes(0) {
    }

    virtual ~allreducef () {
    }

    virtual bool supports_cuda() { return true; }
    
    virtual int64_t report(void *data) {
        return m_totalbytes;
    }

    static void * commFunc(void * args) {
        
        int terminated = 0;

        printf("enterCommThread ");
        int comm_size;
        int rank;
        int comm_step;
        int round_id;
        bool activeNode = false;
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank%2 == 0)
            activeNode = true;

        MPI_Request requests[2];

        //int maxround = (int)floor((log2(comm_size-1)));
        int maxround = (int)floor((log2(comm_size)));
        
        //the active nodes
        if(activeNode){ 
            while(!terminated){
                if(comm_thread_counter == max_int){
                    comm_thread_counter = 0;
                }

                comm_step = (int)(comm_thread_counter / OPS);
                round_id = comm_step % maxround;
                int forward_dst;
                if(round_id == 0)
                    forward_dst = (rank + (int)pow(2, round_id))%comm_size;
                else
                    forward_dst = (rank + (int)pow(2, round_id) + 1)%comm_size;

                //the forward destination
                pthread_spin_lock(&lock[comm_thread_counter%OPS]);
                for(int i=0; i<length[comm_thread_counter%OPS]; i++){
                    input_buff[comm_thread_counter%OPS][i] = shared_model_buff[comm_thread_counter%OPS][i];
                }
                pthread_spin_unlock(&lock[comm_thread_counter%OPS]); 

                MPI_Isend(input_buff[comm_thread_counter%OPS], length[comm_thread_counter%OPS], MPI_FLOAT, forward_dst, comm_thread_counter, MPI_COMM_WORLD, &requests[1]);
                MPI_Irecv(output_buff[comm_thread_counter%OPS], length[comm_thread_counter%OPS], MPI_FLOAT, forward_dst, comm_thread_counter, MPI_COMM_WORLD, &requests[0]);

                MPI_Waitall(2, requests, MPI_STATUSES_IGNORE); 

                //for(int i=0; i<length[comm_thread_counter%OPS]; i++)
                //{
                //    input_buff[comm_thread_counter%OPS][i] = (input_buff[comm_thread_counter%OPS][i] + output_buff[comm_thread_counter%OPS][i])/2;
                //}
  

                pthread_spin_lock(&lock[comm_thread_counter%OPS]);
                for(int i=0; i<length[comm_thread_counter%OPS]; i++)
                {
                    //shared_model_buff[comm_thread_counter%OPS][i] = input_buff[comm_thread_counter%OPS][i];
                    shared_model_buff[comm_thread_counter%OPS][i] = (input_buff[comm_thread_counter%OPS][i] + output_buff[comm_thread_counter%OPS][i])/2;
                }
                pthread_spin_unlock(&lock[comm_thread_counter%OPS]);

                int backward_dst;
                if(round_id == 0)
                    backward_dst = (rank - (int)pow(2, round_id) + comm_size)%comm_size;
                else
                    backward_dst = (rank - (int)pow(2, round_id) - 1 + comm_size)%comm_size;

                //the backward destination
                pthread_spin_lock(&lock[comm_thread_counter%OPS]);
                for(int i=0; i<length[comm_thread_counter%OPS]; i++){
                    input_buff[comm_thread_counter%OPS][i] = shared_model_buff[comm_thread_counter%OPS][i];
                }
                pthread_spin_unlock(&lock[comm_thread_counter%OPS]); 

                MPI_Irecv(output_buff[comm_thread_counter%OPS], length[comm_thread_counter%OPS], MPI_FLOAT, backward_dst, comm_thread_counter, MPI_COMM_WORLD, &requests[0]);
                MPI_Isend(input_buff[comm_thread_counter%OPS], length[comm_thread_counter%OPS], MPI_FLOAT, backward_dst, comm_thread_counter, MPI_COMM_WORLD, &requests[1]);

                MPI_Waitall(2, requests, MPI_STATUSES_IGNORE); 

                //for(int i=0; i<length[comm_thread_counter%OPS]; i++)
                //{
                //    input_buff[comm_thread_counter%OPS][i] = (input_buff[comm_thread_counter%OPS][i] + output_buff[comm_thread_counter%OPS][i])/2;
                //}
  

                pthread_spin_lock(&lock[comm_thread_counter%OPS]);
                for(int i=0; i<length[comm_thread_counter%OPS]; i++)
                {
                    //shared_model_buff[comm_thread_counter%OPS][i] = input_buff[comm_thread_counter%OPS][i];
                    shared_model_buff[comm_thread_counter%OPS][i] = (input_buff[comm_thread_counter%OPS][i] + output_buff[comm_thread_counter%OPS][i])/2;
                }
                pthread_spin_unlock(&lock[comm_thread_counter%OPS]);
                comm_thread_counter++;
            } // endWhile
        } // endIf

        //the passive nodes
        else{
            while(!terminated){
                MPI_Status senderStatus;

                //blocking probe
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &senderStatus);
                int bufferID = senderStatus.MPI_TAG % OPS;

                pthread_spin_lock(&lock[bufferID]);
                for(int i=0; i<length[bufferID]; i++){
                    input_buff[bufferID][i] = shared_model_buff[bufferID][i];
                }
                pthread_spin_unlock(&lock[bufferID]); 

                MPI_Irecv(output_buff[bufferID], length[bufferID], MPI_FLOAT, senderStatus.MPI_SOURCE, senderStatus.MPI_TAG, MPI_COMM_WORLD, &requests[0]);  
                MPI_Isend(input_buff[bufferID], length[bufferID], MPI_FLOAT, senderStatus.MPI_SOURCE, senderStatus.MPI_TAG, MPI_COMM_WORLD, &requests[1]);  
                MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
 
                //for(int i=0; i<length[bufferID]; i++)
                //{
                //    input_buff[bufferID][i] = (input_buff[bufferID][i] + output_buff[bufferID][i])/2;
                //}
  

                pthread_spin_lock(&lock[bufferID]);
                for(int i=0; i<length[bufferID]; i++)
                {
                    //shared_model_buff[bufferID][i] = input_buff[bufferID][i];
                    shared_model_buff[bufferID][i] = (input_buff[bufferID][i] + output_buff[bufferID][i])/2;
                }
                pthread_spin_unlock(&lock[bufferID]);

            } // endWhile
        } // endElse

        return NULL;

    }

    void forward(const float *input, const float*, float *output) {
        
        int comm_size;
        int rank;
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
        if(counter == 0)
        {  
            for(int i=0; i<OPS; i++) 
            {
                output_buff[i] = (float *)calloc(length[i], sizeof(float));
                input_buff[i] = (float *)calloc(length[i], sizeof(float));
                shared_model_buff[i] = (volatile float *)calloc(length[i], sizeof(float));
                pthread_spin_init(&lock[i], PTHREAD_PROCESS_PRIVATE);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if(counter == OPS){
            printf("Pthread Creation");
            int ret;
            ret = pthread_create(&cthread, NULL, commFunc, NULL);
            if (ret){ 
                printf("Pthread creation error ");
            }
        }
        
        pthread_spin_lock(&lock[counter%OPS]);
        if(counter < 2*OPS)
            memcpy(output, input, sizeof(float)*m_len);
        else
            for(int i=0; i<m_len; i++)
                output[i] = (shared_model_buff[counter%OPS][i] + input[i])/2;
        
        for(int i=0; i<m_len; i++)
            shared_model_buff[counter%OPS][i] = output[i];
        pthread_spin_unlock(&lock[counter%OPS]); 


        counter++;
            
    }

    void backward(const float *nextop_grad,
                  const float *fwd_input_tensor,
                  const float*,
                  const float *fwd_output_tensor,
                  float *input_tensor_grad,
                  float *) {
      // Do Nothing here
    }
};


D500_EXPORTED void *create_new_op(deep500::tensor_t *input_descriptors, 
                                  int num_inputs,
                                  deep500::tensor_t *output_descriptors,
                                  int num_outputs) {
   
    int is_init = 0;
    int mt_level;
    MPI_Initialized(&is_init);
    if (!is_init){
        //MPI_Init(NULL, NULL);
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mt_level);
        if (mt_level!=MPI_THREAD_MULTIPLE){
            printf("No MPI_THREAD_MULTIPLE available!  ");
        }
    }

    size_t totalsz = 1;
    for (int i = 0; i < input_descriptors[0].dims; ++i)
        totalsz *= input_descriptors[0].sizes[i];

    return new allreducef(totalsz);
}
D500_REGISTER_OP(allreducef);
"""
