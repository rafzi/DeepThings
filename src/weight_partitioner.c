#include "weight_partitioner.h"
#include "inference_engine_helper.h"
#include <assert.h>


// The extra field is only used in reorg layers. Misuse it to save the original number of filters or channels.
static void backup_orig_n(layer *l)
{
    l->extra = l->n;
}
static void backup_orig_c(layer *l)
{
    l->extra = l->c;
}
static int get_orig_n(layer *l)
{
    return l->extra;
}
static int get_orig_c(layer *l)
{
    return l->extra;
}

static void prune_filters(layer *l, int partition_id, int num_partitions)
{
    int partitionSize = l->n / num_partitions;

    int numFilters = partitionSize;
    if (partition_id == num_partitions - 1)
    {
        // In case the division has a remainder, add the missing filters in the last partition.
        numFilters += l->n % num_partitions;
    }

    // Can use continuous chunk of weight buffer, because the outermost array is filters.
    int filterSize = l->size * l->size * l->c;
    int prunedWeightsSize = filterSize * numFilters;
    float *pruned_weights = malloc(prunedWeightsSize * sizeof(float));
    memcpy(pruned_weights, l->weights + partition_id * filterSize * partitionSize, prunedWeightsSize * sizeof(float));

    free(l->weights);
    l->weights = pruned_weights;

    backup_orig_n(l);

    int outSize = l->w * l->h * numFilters;
    l->outputs = outSize;
    l->out_c = numFilters;
    l->n = numFilters;
    l->scales += partition_id * partitionSize;
    l->biases += partition_id * partitionSize;
    l->rolling_mean += partition_id * partitionSize;
    l->rolling_variance += partition_id * partitionSize;
}

static void prune_channels(layer *l, int partition_id, int num_partitions)
{
    int partitionSize = l->c / num_partitions;

    int numChannels = partitionSize;
    if (partition_id == num_partitions - 1)
    {
        // In case the division has a remainder, add the missing channels in the last partition.
        numChannels += l->c % num_partitions;
    }

    int filterSize = l->size * l->size * l->c;
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

    backup_orig_c(l);

    l->c = numChannels;
}


bool is_weight_part_fused_layer(cnn_model *model, int layer_id)
{
    for (int i = 0; i < model->weight_part_para.num_fused_layers; i++)
    {
        if (model->weight_part_para.fused_layers[i] == layer_id)
        {
            return true;
        }
    }

    return false;
}

bool is_entire_weightpart_input_required(cnn_model *model, int layer_id)
{
    if (layer_id <= model->ftp_para->fused_layers)
    {
        return true;
    }
    network *net = model->net;
    if (is_weight_part_fused_layer(model, layer_id - 1) || is_weight_part_fused_layer(model, layer_id - 2))
    {
        return true;
    }
    layer *prev_l = &net->layers[layer_id - 1];
    return prev_l->type != CONVOLUTIONAL;
}

void load_partitioned_weights(cnn_model *model, int32_t cli_id, int num_partitions)
{
    // FIXME: this is just pruning weights that have already been loaded. it should only load the neccessary weights in
    // the first place.

    assert(model->ftp_para->fused_layers != 0);

    network *net = model->net;
    weight_partitioning_parameters *para = &model->weight_part_para;

    para->first_partitioned_layer = 0;
    para->num_part_layers = 0;
    para->num_fused_layers = 0;

    // TODO: for now this simply gives each client a weight partition.
    int partition_id = cli_id % num_partitions;

    // Go through the remaining layers after FTP.
    for (int i = model->ftp_para->fused_layers; i < net->n; i++)
    {
        layer *l = &net->layers[i];

        if (l->type != CONVOLUTIONAL)
        {
            continue;
        }

        // Record the first partitioned layer.
        if (para->first_partitioned_layer == 0)
        {
            para->first_partitioned_layer = i;
        }

        para->partitioned_layers[para->num_part_layers] = i;
        para->num_part_layers++;

        prune_filters(l, partition_id, num_partitions);

        // continue; /// SKIP FUSING

        int next_i = i + 1;
        layer *next_l = &net->layers[next_i];
        if (next_i < net->n && next_l->type == CONVOLUTIONAL)
        {
            // We can fuse layers.
            para->fused_layers[para->num_fused_layers] = i;
            para->num_fused_layers++;

            prune_channels(next_l, partition_id, num_partitions);
            i++;
        }
    }
}

/*int get_weight_part_output_offset(layer *l, int partition_id, int num_partitions)
{
    int partitionSize = l->n / num_partitions;
    return partition_id * l->w * l->h * partition_size * sizeof(float);
}*/

void copy_weight_part_output(layer *l, float *data, int partition_id, int num_partitions)
{
    int orig_num_filters = get_orig_n(l);
    int partition_size = orig_num_filters / num_partitions;
    int num_filters = partition_size;
    if (partition_id == num_partitions - 1)
    {
        num_filters += orig_num_filters % num_partitions;
    }

    int out_offset = partition_id * l->w * l->h * partition_size;
    int out_size = l->w * l->h * num_filters;

    memcpy(l->output + out_offset, data, out_size * sizeof(float));
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
