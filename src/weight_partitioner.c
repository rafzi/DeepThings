#include "weight_partitioner.h"
#include "inference_engine_helper.h"
#include <assert.h>


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

    // This field is only used in reorg layers. Misuse it to save the original number of filters.
    l->extra = l->n;

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
        int weightOffset = i * filterSize * partition_id * filterSizeReorgPrev;
        memcpy(reorg_weights + i * filterSizeReorg, l->weights + weightOffset, filterSizeReorg * sizeof(float));
    }

    free(l->weights);
    l->weights = reorg_weights;

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
        printf("layer %d: orignumfilt: %d, numfilt: %d\n", i, l->extra, l->n);

        continue; /// SKIP FUSING FOR NOW

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
    int orig_num_filters = l->extra;
    int partition_size = orig_num_filters / num_partitions;
    int num_filters = partition_size;
    if (partition_id == num_partitions - 1)
    {
        num_filters += orig_num_filters % num_partitions;
    }

    printf("restoring tiles: orignumfilt: %d, numfilt: %d, for this(%d): %d\n", orig_num_filters, l->n, partition_id, num_filters);

    int out_offset = partition_id * l->w * l->h * partition_size;
    int out_size = l->w * l->h * num_filters;

    printf("out offset: %d,  out_sz: %d\n", out_offset, out_size);

    memcpy(l->output + out_offset, data, out_size * sizeof(float));
}
