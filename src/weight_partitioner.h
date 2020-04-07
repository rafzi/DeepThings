#ifndef WEIGHT_PARTITIONER_H
#define WEIGHT_PARTITIONER_H

#include "configure.h"
#include "global_context.h"
#include "darknet.h"
#include <stdint.h>
#include <stdbool.h>


struct cnn_model_wrapper;
typedef struct cnn_model_wrapper cnn_model;


enum layer_partition_type
{
    LAYER_PART_TYPE_NONE,
    LAYER_PART_TYPE_LOP,
    LAYER_PART_TYPE_LIP,
    LAYER_PART_TYPE_FUSE1,
    LAYER_PART_TYPE_FUSE2
};

typedef struct
{
    enum layer_partition_type *type;
} weight_partitioning_parameters;


bool is_weight_part_fused_layer(cnn_model *model, int layer_id);
bool is_lip_layer(cnn_model *model, int layer_id);
bool can_reuse_lop_output(cnn_model *model, int layer_id);

void load_partitioned_weights(cnn_model *model, int32_t cli_id, int num_partitions);

int get_lop_input_offset(layer *l, int partition_id, int num_partitions);
size_t get_lop_input_size(layer *l, int partition_id, int num_partitions);
int get_lop_weight_offset(layer *l, int partition_id, int num_partitions);
size_t get_lop_weight_size(layer *l, int partition_id, int num_partitions);
int get_lop_output_offset(layer *l, int partition_id, int num_partitions);

void copy_weight_part_output(layer *l, float *data, int partition_id, int num_partitions);
void finalize_weight_part_fused_output(layer *l, network *net);

const char *get_layer_type_name(enum layer_partition_type type);

#endif
