#ifndef WEIGHT_PARTITIONER_H
#define WEIGHT_PARTITIONER_H

#include "configure.h"
#include <stdint.h>


#define MAX_PARTITIONED_WEIGHT_LAYERS 32


struct cnn_model_wrapper;
typedef struct cnn_model_wrapper cnn_model;


typedef struct
{
    int first_partitioned_layer;
    int partitioned_layers[MAX_PARTITIONED_WEIGHT_LAYERS];
    int num_part_layers;
    int fused_layers[MAX_PARTITIONED_WEIGHT_LAYERS];
    int num_fused_layers;
} weight_partitioning_parameters;


void load_partitioned_weights(cnn_model *model, int32_t cli_id, int num_partitions);

#endif
