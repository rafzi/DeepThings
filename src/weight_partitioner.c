#include "weight_partitioner.h"
#include "inference_engine_helper.h"
#include <assert.h>

// The extra and flatten fields are only used in reorg layers. Misuse them to save the original number of filters or channels.
static void backup_orig_n_c(layer *l)
{
    l->extra = l->n;
    l->flatten = l->c;
}
static int get_orig_n(layer *l)
{
    return l->extra;
}
static int get_orig_c(layer *l)
{
    return l->flatten;
}

static int get_partition_size_filters(layer *l, int num_partitions)
{
    return get_orig_n(l) / num_partitions;
}
static int get_partition_size_channels(layer *l, int num_partitions)
{
    return get_orig_c(l) / num_partitions;
}
static int get_num_filters(layer *l, int partition_id, int num_partitions)
{
    int numFilters = get_partition_size_filters(l, num_partitions);
    if (partition_id == num_partitions - 1)
    {
        // In case the division has a remainder, add the missing filters in the last partition.
        numFilters += get_orig_n(l) % num_partitions;
    }
    return numFilters;
}
static int get_num_channels(layer *l, int partition_id, int num_partitions)
{
    int numChannels = get_partition_size_channels(l, num_partitions);
    if (partition_id == num_partitions - 1)
    {
        // In case the division has a remainder, add the missing channels in the last partition.
        numChannels += get_orig_c(l) % num_partitions;
    }
    return numChannels;
}
static int get_filter_size(layer *l)
{
    return l->size * l->size * l->c;
}


static void prune_filters(layer *l, int partition_id, int num_partitions)
{
    // Can use continuous chunk of weight buffer, because the outermost array is filters.
    int w_offset = get_lop_weight_offset(l, partition_id, num_partitions);
    size_t w_size = get_lop_weight_size(l, partition_id, num_partitions);
    float *pruned_weights = malloc(w_size);
    memcpy(pruned_weights, l->weights + w_offset, w_size);

    free(l->weights);
    l->weights = pruned_weights;

    int numFilters = get_num_filters(l, partition_id, num_partitions);
    int data_offset = partition_id * get_partition_size_filters(l, num_partitions);

    l->outputs = l->w * l->h * numFilters;
    l->out_c = numFilters;
    l->n = numFilters;
    l->scales += data_offset;
    l->biases += data_offset;
    l->rolling_mean += data_offset;
    l->rolling_variance += data_offset;
}

static void prune_channels(layer *l, int partition_id, int num_partitions)
{
    int partitionSize = get_partition_size_channels(l, num_partitions);
    int numChannels = get_num_channels(l, partition_id, num_partitions);

    int filterSize = get_filter_size(l);
    int filterSizeReorgPrev = l->size * l->size * partitionSize;
    int filterSizeReorg = l->size * l->size * numChannels;

    // The weights need to be continuous for nnp. Since we only use a subset of channels in every filter, move them
    // together.
    float *reorg_weights = malloc(l->n * filterSizeReorg * sizeof(float));
    for (int i = 0; i < l->n; i++)
    {
        int weightOffset = i * filterSize + partition_id * filterSizeReorgPrev;
        memcpy(reorg_weights + i * filterSizeReorg, l->weights + weightOffset, filterSizeReorg * sizeof(float));
    }

    free(l->weights);
    l->weights = reorg_weights;

    l->c = numChannels;
}

bool is_weight_part_fused_layer(cnn_model *model, int layer_id)
{
    return model->weight_part_para.type[layer_id] == LAYER_PART_TYPE_FUSE1;
}

bool is_lip_layer(cnn_model *model, int layer_id)
{
    return model->weight_part_para.type[layer_id] == LAYER_PART_TYPE_LIP;
}

bool can_reuse_lop_output(cnn_model *model, int layer_id)
{
    return model->weight_part_para.type[layer_id - 1] == LAYER_PART_TYPE_LOP &&
           (model->weight_part_para.type[layer_id] == LAYER_PART_TYPE_LOP ||
            model->weight_part_para.type[layer_id] == LAYER_PART_TYPE_FUSE1);
}

void load_partitioned_weights(cnn_model *model, int32_t cli_id, int num_partitions)
{
    // FIXME: this is just pruning weights that have already been loaded. it should only load the neccessary weights in
    // the first place.
    // parser.c:load_convolutional_weights
    // That function and its parents would have to be extended by
    // additional args to figure out the network configuration.

    assert(model->ftp_para->fused_layers != 0);

    network *net = model->net;
    weight_partitioning_parameters *para = &model->weight_part_para;

    // TODO: for now this simply gives each client a weight partition.
    int partition_id = cli_id % num_partitions;

    for (int i = 0; i < net->n; i++)
    {
        layer *l = &net->layers[i];
        if (para->type[i] == LAYER_PART_TYPE_NONE)
        {
            continue;
        }

        if (i < model->ftp_para->fused_layers)
        {
            printf("Invalid layer type: Partitioned must be after FTP\n");
            exit(1);
        }
        if (l->type != CONVOLUTIONAL)
        {
            printf("Invalid layer type: Partitioned must be convolutional\n");
            exit(1);
        }

        backup_orig_n_c(l);

        switch (para->type[i])
        {
        case LAYER_PART_TYPE_FUSE1:
            if (i + 1 >= net->n || para->type[i+1] != LAYER_PART_TYPE_FUSE2)
            {
                printf("Invalid layer type: F1 must be followed by F2\n");
                exit(1);
            }
            // fallthrough
        case LAYER_PART_TYPE_LOP:
            prune_filters(l, partition_id, num_partitions);
            break;
        case LAYER_PART_TYPE_FUSE2:
            if (i - 1 < 0 || para->type[i-1] != LAYER_PART_TYPE_FUSE1)
            {
                printf("Invalid layer type: F2 must be preceded by F1\n");
                exit(1);
            }
            // fallthrough
        case LAYER_PART_TYPE_LIP:
            prune_channels(l, partition_id, num_partitions);
            break;
        default:
            printf("Invalid layer type: Convolutional must be partitioned\n");
            exit(1);
        }
    }
}


int get_lop_input_offset(layer *l, int partition_id, int num_partitions)
{
    return partition_id * l->w * l->h * get_partition_size_channels(l, num_partitions);
}

size_t get_lop_input_size(layer *l, int partition_id, int num_partitions)
{
    return l->w * l->h * get_num_channels(l, partition_id, num_partitions) * sizeof(float);
}

int get_lop_weight_offset(layer *l, int partition_id, int num_partitions)
{
    return partition_id * get_filter_size(l) * get_partition_size_filters(l, num_partitions);
}

size_t get_lop_weight_size(layer *l, int partition_id, int num_partitions)
{
    return get_filter_size(l) * get_num_filters(l, partition_id, num_partitions) * sizeof(float);
}

int get_lop_output_offset(layer *l, int partition_id, int num_partitions)
{
    return partition_id * l->w * l->h * get_partition_size_filters(l, num_partitions);
}


void copy_weight_part_output(layer *l, float *data, int partition_id, int num_partitions)
{
    int out_offset = get_lop_output_offset(l, partition_id, num_partitions);
    size_t out_size = l->outputs * sizeof(float);
    memcpy(l->output + out_offset, data, out_size);
}

void finalize_weight_part_fused_output(layer *l, network *net)
{
    // Finalize the layer.
    int out_h = convolutional_out_height(*l);
    int out_w = convolutional_out_width(*l);
    int n = out_h * out_w;

    if (l->batch_normalize)
    {
        forward_batchnorm_layer(*l, *net);
    }
    else
    {
        add_bias(l->output, l->biases, l->batch, l->n, out_h * out_w);
    }

    activate_array_thread(l->output, l->n, n, l->activation, net->threadpool);
    if (l->binary || l->xnor)
        swap_binary(l);
}

const char *get_layer_type_name(enum layer_partition_type type)
{
    switch (type)
    {
    case LAYER_PART_TYPE_NONE:
        return "Not partitioned";
    case LAYER_PART_TYPE_LOP:
        return "LOP";
    case LAYER_PART_TYPE_LIP:
        return "LIP";
    case LAYER_PART_TYPE_FUSE1:
        return "FUSE1";
    case LAYER_PART_TYPE_FUSE2:
        return "FUSE2";
    default:
        return "Unknown";
    }
}
