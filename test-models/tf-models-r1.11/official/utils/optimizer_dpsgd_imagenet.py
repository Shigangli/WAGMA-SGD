
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
                #model_avg = op(var, last_var)

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
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>


int counter = 0;

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
float* grad_ptrs_right[OPS];
float* grad_ptrs_left[OPS];

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

    
    void forward(const float *input, const float*, float *output) {

    int neighborRight;
    int neighborLeft;
    MPI_Request requests[2];
    MPI_Status  status;
    
    int comm_size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(counter == 0)
    {  
        for(int i=0; i<OPS; i++) 
        {
            grad_ptrs_right[i] = (float *)calloc(length[i], sizeof(float));
            grad_ptrs_left[i] = (float *)calloc(length[i], sizeof(float));
        }
    }

    neighborRight = (rank+1)%comm_size;
    neighborLeft = (rank-1+comm_size)%comm_size;

    MPI_Irecv(grad_ptrs_right[counter%OPS], m_len, MPI_FLOAT, neighborLeft, counter, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(grad_ptrs_left[counter%OPS], m_len, MPI_FLOAT, neighborRight, counter, MPI_COMM_WORLD, &requests[1]);
    MPI_Send(input, m_len, MPI_FLOAT, neighborRight, counter, MPI_COMM_WORLD);
    MPI_Send(input, m_len, MPI_FLOAT, neighborLeft, counter, MPI_COMM_WORLD);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

    for(int i=0; i<m_len; i++)
    {
        output[i] = (input[i] + grad_ptrs_right[counter%OPS][i] + grad_ptrs_left[counter%OPS][i])/3;
    }  

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
    MPI_Initialized(&is_init);
    if (!is_init)
        MPI_Init(NULL, NULL);

    size_t totalsz = 1;
    for (int i = 0; i < input_descriptors[0].dims; ++i)
        totalsz *= input_descriptors[0].sizes[i];

    return new allreducef(totalsz);
}
D500_REGISTER_OP(allreducef);
"""
