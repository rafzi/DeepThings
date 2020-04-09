#include <stdio.h>
#include "deepthings_edge.h"
#include "ftp.h"
#include "inference_engine_helper.h"
#include "frame_partitioner.h"
#include "reuse_data_serialization.h"
#if DEBUG_TIMING
static double start_time;
static double acc_time[MAX_EDGE_NUM];
static uint32_t acc_frames[MAX_EDGE_NUM];
#endif
#if DEBUG_COMMU_SIZE
static double commu_size;
#endif

static void set_lt(enum layer_partition_type *p, int i, enum layer_partition_type type, int fused_layers)
{
   if (i >= fused_layers)
   {
      p[i] = type;
   }
}

device_ctxt* deepthings_edge_init(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t cli_num, const char** edge_addr_list){
   device_ctxt* ctxt = init_client(edge_id, FRAME_NUM);
   cnn_model* model = load_cnn_model(network, weights);
   model->ftp_para = preform_ftp(N, M, fused_layers, model->net_para);
#if DATA_REUSE
   model->ftp_para_reuse = preform_ftp_reuse(model->net_para, model->ftp_para);
#endif
   ctxt->model = model;

   enum layer_partition_type *lt = model->weight_part_para.type;
#ifdef SKIP_FUSING
#pragma message("FUSION WILL BE SKIPPED")
   for (int i = 0; i < model->net->n; i++)
   {
      layer *l = &model->net->layers[i];
      if (l->type == CONVOLUTIONAL && i >= fused_layers)
      {
         lt[i] = LAYER_PART_TYPE_LOP;
      }
   }
#else
   set_lt(lt, 16, LAYER_PART_TYPE_LOP, fused_layers);
   set_lt(lt, 18, LAYER_PART_TYPE_FUSE1, fused_layers);
   set_lt(lt, 19, LAYER_PART_TYPE_FUSE2, fused_layers);
   set_lt(lt, 20, LAYER_PART_TYPE_FUSE1, fused_layers);
   set_lt(lt, 21, LAYER_PART_TYPE_FUSE2, fused_layers);
   set_lt(lt, 22, LAYER_PART_TYPE_LOP, fused_layers);
   set_lt(lt, 23, LAYER_PART_TYPE_FUSE1, fused_layers);
   set_lt(lt, 24, LAYER_PART_TYPE_FUSE2, fused_layers);
   set_lt(lt, 26, LAYER_PART_TYPE_LIP, fused_layers);
   set_lt(lt, 29, LAYER_PART_TYPE_FUSE1, fused_layers);
   set_lt(lt, 30, LAYER_PART_TYPE_FUSE2, fused_layers);
#endif

   load_partitioned_weights(model, edge_id, cli_num);

   for (int i = 0; i < model->net->n; i++)
   {
      printf("layer %i has type: %s\n", i, get_layer_type_name(model->weight_part_para.type[i]));
   }

   set_gateway_local_addr(ctxt, GATEWAY_LOCAL_ADDR);
   set_gateway_public_addr(ctxt, GATEWAY_PUBLIC_ADDR);
   set_total_frames(ctxt, FRAME_NUM);
   set_batch_size(ctxt, N*M);

#ifdef NNPACK
   nnp_initialize();
   model->net->threadpool = pthreadpool_create(THREAD_NUM);
#endif

   // FIXME: issue: edge needs to know which other edge to send
   // the results to based on cli_id of a task. this is a bad
   // solution. rather, put client address in
   // meta annotation when receiving task. this is the addres to send
   // the results back to.
   ctxt->total_cli_num = cli_num;
   ctxt->addr_list = (char**)malloc(sizeof(char*)*cli_num);
   for(uint32_t i = 0; i < cli_num; i++){
      ctxt->addr_list[i] = (char*)malloc(sizeof(char)*ADDR_LEN);
      strcpy(ctxt->addr_list[i], edge_addr_list[i]);
   }

   // Weight partitioning part.
   ctxt->result_queue_weightpart = new_queue(MAX_QUEUE_SIZE);
   ctxt->ready_pool_weightpart = new_queue(MAX_QUEUE_SIZE);
   ctxt->results_pool_weightpart = new_queue(MAX_QUEUE_SIZE);
   ctxt->task_queue_weightpart = malloc(ctxt->total_cli_num * sizeof(thread_safe_queue*));
   for (int i = 0; i < ctxt->total_cli_num; i++)
   {
      ctxt->task_queue_weightpart[i] = new_queue(MAX_QUEUE_SIZE);
   }

   return ctxt;
}

#if DATA_REUSE
void send_reuse_data(device_ctxt* ctxt, blob* task_input_blob){
   cnn_model* model = (cnn_model*)(ctxt->model);
   /*if task doesn't generate any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 1) return;

   service_conn* conn;

   blob* temp  = self_reuse_data_serialization(ctxt, get_blob_task_id(task_input_blob), get_blob_frame_seq(task_input_blob));
   conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT);
   send_request("reuse_data", 20, conn);
#if DEBUG_DEEP_EDGE
   printf("send self reuse data for task %d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob));
#endif
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);
   close_service_connection(conn);
}

void request_reuse_data(device_ctxt* ctxt, blob* task_input_blob, bool* reuse_data_is_required){
   cnn_model* model = (cnn_model*)(ctxt->model);
   /*if task doesn't require any reuse_data*/
   if(model->ftp_para_reuse->schedule[get_blob_task_id(task_input_blob)] == 0) return;/*Task without any dependency*/
   if(!need_reuse_data_from_gateway(reuse_data_is_required)) return;/*Reuse data are all generated locally*/

   service_conn* conn;
   conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT);
   send_request("request_reuse_data", 20, conn);
   char data[20]="empty";
   blob* temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), 20, (uint8_t*)data);
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);
#if DEBUG_DEEP_EDGE
   printf("Request reuse data for task %d:%d \n", get_blob_cli_id(task_input_blob), get_blob_task_id(task_input_blob));
#endif

   temp = new_blob_and_copy_data(get_blob_task_id(task_input_blob), sizeof(bool)*4, (uint8_t*)reuse_data_is_required);
   copy_blob_meta(temp, task_input_blob);
   send_data(temp, conn);
   free_blob(temp);


   temp = recv_data(conn);
   copy_blob_meta(temp, task_input_blob);
   overlapped_tile_data** temp_region_and_data = adjacent_reuse_data_deserialization(model, get_blob_task_id(temp), (float*)temp->data, get_blob_frame_seq(temp), reuse_data_is_required);
   place_adjacent_deserialized_data(model, get_blob_task_id(temp), temp_region_and_data, reuse_data_is_required);
   free_blob(temp);

   close_service_connection(conn);
}
#endif

static inline void process_task(device_ctxt* ctxt, blob* temp, bool is_reuse){
   cnn_model* model = (cnn_model*)(ctxt->model);
   blob* result;
   set_model_input(model, (float*)temp->data);
   forward_partition(model, get_blob_task_id(temp), is_reuse);
   result = new_blob_and_copy_data(0,
                                      get_model_byte_size(model, model->ftp_para->fused_layers-1),
                                      (uint8_t*)(get_model_output(model, model->ftp_para->fused_layers-1))
                                     );
#if DATA_REUSE
   send_reuse_data(ctxt, temp);
#endif
   copy_blob_meta(result, temp);
   enqueue(ctxt->result_queue, result);
   free_blob(result);

}

static blob *process_task_weightpart(device_ctxt *ctxt, blob *task){
   cnn_model *model = (cnn_model*)ctxt->model;

   set_model_input(model, (float*)task->data);

   int layer_id = task->id;
   layer *l = &model->net->layers[layer_id];
   bool is_lip = is_lip_layer(model, layer_id);
   bool is_fused = is_weight_part_fused_layer(model, layer_id);

   if (!is_lip)
   {
   forward_convolutional_layer_nnpack(*l, *model->net);
   }
   if (is_fused)
   {
      model->net->input = l->output;
      if (l->truth)
      {
         model->net->truth = l->output;
      }

      layer_id++;
   }

   // If fused, process the next layer.
   if (is_fused || is_lip)
   {
      l = &model->net->layers[layer_id];

      model->net->index = layer_id;
      if (l->delta)
      {
         fill_cpu(l->outputs * l->batch, 0, l->delta, 1);
      }

      struct nnp_size input_size = { l->w, l->h };
      struct nnp_padding input_padding = { l->pad, l->pad, l->pad, l->pad };
      struct nnp_size kernel_size = { l->size, l->size };
      struct nnp_size stride = { l->stride, l->stride };

      nnp_convolution_inference(nnp_convolution_algorithm_implicit_gemm, nnp_convolution_transform_strategy_tuple_based,
                              l->c, l->n, input_size, input_padding, kernel_size, stride, model->net->input,
                              l->weights, NULL, l->output, NULL, NULL, nnp_activation_identity, NULL,
                              model->net->threadpool, NULL);
   }

   blob *result = new_blob_and_copy_data(0, get_model_byte_size(model, layer_id), (uint8_t*)get_model_output(model, layer_id));
   copy_blob_meta(result, task);
   annotate_blob(result, get_this_client_id(ctxt), 0, task->id);
   result->id = task->id;
   return result;
}

void partition_frame_and_perform_inference_thread(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* model = (cnn_model*)(ctxt->model);
   blob* temp = NULL;
   uint32_t frame_num;
   bool* reuse_data_is_required;

   uint32_t time_start = sys_now();

   int32_t own_cli_id = get_this_client_id(ctxt);

   for(frame_num = 0; frame_num < FRAME_NUM; frame_num ++){
      /*Wait for i/o device input*/
      /*recv_img()*/

      /*Load image and partition, fill task queues*/
      load_image_as_model_input(model, frame_num);
      register_client(ctxt);

      int first_conv_layer = model->net_para->first_conv_layer;
      // Process any potential preprocessing layers before convs start.
      for (int i = 0; i < first_conv_layer; i++)
      {
         network *net = model->net;
         layer *l = &net->layers[i];
         l->forward(*l, *net);
      }

      image_holder img;
      int32_t cli_id;
      int32_t frame_seq;

      if (model->ftp_para->fused_layers <= first_conv_layer) {
         img = load_image_as_model_input(model, frame_num);
         cli_id = own_cli_id;
         frame_seq = frame_num;
      } else {
         partition_and_enqueue(ctxt, frame_num);

         /*Dequeue and process task*/
         while(1){
            temp = try_dequeue(ctxt->task_queue);
            if(temp == NULL) break;
            bool data_ready = false;
#if DEBUG_DEEP_EDGE
            printf("====================Processing task id is %d, data source is %d, frame_seq is %d====================\n", get_blob_task_id(temp), get_blob_cli_id(temp), get_blob_frame_seq(temp));
#endif/*DEBUG_DEEP_EDGE*/
#if DATA_REUSE
            data_ready = is_reuse_ready(model->ftp_para_reuse, get_blob_task_id(temp));
            if((model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) && data_ready) {
               blob* shrinked_temp = new_blob_and_copy_data(get_blob_task_id(temp),
                        (model->ftp_para_reuse->shrinked_input_size[get_blob_task_id(temp)]),
                        (uint8_t*)(model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
               copy_blob_meta(shrinked_temp, temp);
               free_blob(temp);
               temp = shrinked_temp;


               reuse_data_is_required = check_missing_coverage(model, get_blob_task_id(temp), get_blob_frame_seq(temp));
#if DEBUG_DEEP_EDGE
               printf("Request data from gateway, is there anything missing locally? ...\n");
               print_reuse_data_is_required(reuse_data_is_required);
#endif/*DEBUG_DEEP_EDGE*/
               request_reuse_data(ctxt, temp, reuse_data_is_required);
               free(reuse_data_is_required);
            }
#if DEBUG_DEEP_EDGE
            if((model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1) && (!data_ready))
               printf("The reuse data is not ready yet!\n");
#endif/*DEBUG_DEEP_EDGE*/

#endif/*DATA_REUSE*/
            process_task(ctxt, temp, data_ready);
            free_blob(temp);
#if DEBUG_COMMU_SIZE
            printf("======Communication size at edge is: %f======\n", ((double)commu_size)/(1024.0*1024.0*FRAME_NUM));
#endif
         }


         // Now wait for stealers to return their results.
         temp = dequeue_and_merge(ctxt);
         cli_id = get_blob_cli_id(temp);
         frame_seq = get_blob_frame_seq(temp);
   #if DEBUG_FLAG
         printf("Client %d, frame sequence number %d, all partitions are merged in deepthings_merge_result_thread\n", cli_id, frame_seq);
   #endif
         float* fused_output = (float*)(temp->data);
         img = load_image_as_model_input(model, get_blob_frame_seq(temp));
         set_model_input(model, fused_output);
      }

      network *net = model->net;
      int opfd_start = model->ftp_para->fused_layers > first_conv_layer ? model->ftp_para->fused_layers : first_conv_layer;
      for (int i = opfd_start; i < net->n; i++){
         //printf("===weight part: layer %d/%d\n", i, net->n - 1);
printf("start layer %i at %d\n", i, sys_now() - time_start);
         layer *l = &net->layers[i];
         net->index = i;
         if (l->delta){
            fill_cpu(l->outputs * l->batch, 0, l->delta, 1);
         }

         if (l->type == CONVOLUTIONAL){
            // Is a distributed layer.

            blob *task_input = new_blob_and_move_data(i, l->inputs * sizeof(float), net->input);

            // Create blobs for each partial result of the previous layer.
            // Clients will not need the data of their partition if they calculated it before.
            // See weight_partitioner.c:prune_filters()
            int num_parts = ctxt->total_cli_num;
            blob **task_inputs = malloc(num_parts * sizeof(blob*));
            for (int c = 0; c < num_parts; c++){
               int input_offset = get_lop_input_offset(l, c, num_parts);
               size_t input_size = get_lop_input_size(l, c, num_parts);
               task_inputs[c] = new_blob_and_copy_data(i, input_size, net->input + input_offset);
            }
            blob *dummy = new_blob_and_alloc_data(i, 1);

            // into task queue? no, need id specific
            if (is_lip_layer(model, i))
            {
               for (int target_cli_id = 0; target_cli_id < ctxt->total_cli_num; target_cli_id++)
               {
                  enqueue(ctxt->task_queue_weightpart[target_cli_id], task_inputs[target_cli_id]);
               }
            }
            else
            {
               for (int target_cli_id = 0; target_cli_id < ctxt->total_cli_num; target_cli_id++){
                  if (target_cli_id == ctxt->this_cli_id){
                     enqueue(ctxt->task_queue_weightpart[target_cli_id], task_input);
                  } else {
                     for (int c = 0; c < ctxt->total_cli_num; c++){
                        if (can_reuse_lop_output(model, i) &&
                           c == target_cli_id){
                           // Target cli can reuse data. Just queue a dummy segment.
                           enqueue(ctxt->task_queue_weightpart[target_cli_id], dummy);
                        } else {
                           enqueue(ctxt->task_queue_weightpart[target_cli_id], task_inputs[c]);
                        }
                     }
                  }
               }
            }
printf("before processing: %d\n", sys_now() - time_start);

            // Process local tasks.
            while (1)
            {
               blob *task_wpart = try_dequeue(ctxt->task_queue_weightpart[own_cli_id]);
               if (!task_wpart){
                  //printf("===weight part: no more local tasks for me\n");
                  break;
               }

               //printf("===weight part: processing local task %d\n", task_wpart->id);

               blob *result = process_task_weightpart(ctxt, task_wpart);
               free_blob(task_wpart);
               store_weight_part_result(ctxt, result);
               free_blob(result);
            }

            // Wait for results.
            //printf("===weight part: waiting for other results\n");
            blob *ready = dequeue(ctxt->ready_pool_weightpart);
printf("results ready: %d\n", sys_now() - time_start);

            // Only now we can free the input.
            free_blob(task_input);
            for (int c = 0; c < ctxt->total_cli_num; c++){
               free_blob(task_inputs[c]);
            }
            free(task_inputs);
            free_blob(dummy);

            // Merge outputs.
            int num_partitions = ctxt->total_cli_num;
            if (is_weight_part_fused_layer(model, i) || is_lip_layer(model, i))
            {
               // More elaborate merging: add outputs and finalize layer.
printf("1: %d\n", sys_now() - time_start);
               if (is_weight_part_fused_layer(model, i))
               {
               // Advance one layer.
               i++;
                  //printf("##### processing output of merged layer %d + %d\n", i - 1, i);
               }
               l = &net->layers[i];
               net->index = i;
               if (l->delta){
                  fill_cpu(l->outputs * l->batch, 0, l->delta, 1);
               }

               // Need to set to zero for fused layers, because results will be added on top.
               memset(l->output, 0, l->outputs * sizeof(float));
printf("3: %d\n", sys_now() - time_start);

               for (int j = 0; j < num_partitions; j++)
               {
                  blob *result = dequeue(ctxt->results_pool_weightpart);
                  int layer_id = result->id;
                  if (layer_id != (is_lip_layer(model, i) ? i : i - 1))
                  {
                     printf("ERROR: got unexpected layer id!\n");
                     exit(1);
                  }

                  //print_part_data(result->data, l->outputs);

                  // Accumulate to the output.
                  // TODO is there an nnpack accelerator for this?
                  for (int k = 0; k < l->outputs; k++)
                  {
                     l->output[k] += ((float*)result->data)[k];
                  }
               }
printf("4: %d\n", sys_now() - time_start);

               finalize_weight_part_fused_output(l, net);
printf("5: %d\n", sys_now() - time_start);
            } else {
               // Simple concatenation of outputs.

               for (int j = 0; j < num_partitions; j++)
               {
                  blob *result = dequeue(ctxt->results_pool_weightpart);
                  int layer_id = result->id;
                  if (layer_id != i)
                  {
                     printf("ERROR: got unexpected layer id!\n");
                  }

                  int partition_id = get_blob_cli_id(result);
                  copy_weight_part_output(l, result->data, partition_id, num_partitions);
               }
            }
         }else{
            // Not convolutional. Execute locally.
            l->forward(*l, *net);
         }

         //print_out(l);

         net->input = l->output;
         if (l->truth){
            net->truth = l->output;
         }
      }

      //forward_all(model, model->ftp_para->fused_layers);
      printf("saving frame: %d\n", frame_seq);
      draw_object_boxes(model, frame_seq);
      free_image_holder(model, img);
      if (temp) {
         free_blob(temp);
      }
#if DEBUG_FLAG
      printf("Client %d, frame sequence number %d, finish processing\n", cli_id, frame_seq);
#endif

      if (net->layers[net->n - 1].type == SOFTMAX) {
         int predictions[5];
         top_predictions(net, 5, predictions);
         printf("top predictions:\n");
         for (int i = 0; i < 5; i++) {
            printf("  %i\n", predictions[i]);
         }
      }

      /*Unregister and prepare for next image*/
      cancel_client(ctxt);
   }
   uint32_t run_time = sys_now() - time_start;
   printf("Finished in %d ms\n", run_time);

   FILE *procStatus = fopen("/proc/self/status", "r");
   int maxVirtMem = 0, maxRealMem = 0;
   char *buf = NULL;
   size_t sz = 0;
   while (getline(&buf, &sz, procStatus) >= 0)
   {
      if (strncmp(buf, "VmPeak", 6) == 0)
      {
         sscanf(buf, "VmPeak: %d", &maxVirtMem);
      }
      else
      {
         sscanf(buf, "VmHWM: %d", &maxRealMem);
      }
   }
   free(buf);
   fclose(procStatus);

   FILE *out_file;
   out_file = fopen("result_times.txt", "a");
   fprintf(out_file, "%d %d %d %d %s\n", run_time, MAX_EDGE_NUM, maxVirtMem, maxRealMem,
#ifdef SKIP_FUSING
	"skip"
#else
	"no-skip"
#endif
		   );
   fclose(out_file);
}


typedef struct
{
   int32_t steal_from_cli_id;
   device_ctxt *ctxt;
   bool *done; // FIXME: must be atomic.
} steal_weightpart_args;

void steal_weightpart_thread(void *arg){
   steal_weightpart_args *args = (steal_weightpart_args*)arg;
   device_ctxt *ctxt = args->ctxt;

   service_conn *conn = connect_service(TCP, get_client_addr(args->steal_from_cli_id, ctxt), WEIGHT_PART_PORT);

static int total_recv = 0;
static int total_send = 0;

   // Keep receiving data.
   while (1)
   {
uint32_t time_start = sys_now();

      blob **tasks = malloc(ctxt->total_cli_num * sizeof(blob*));
      tasks[0] = recv_data(conn);
      total_recv += tasks[0]->size;
      if (tasks[0]->id == -1)
      {
         // No more tasks with this client. Close the connection.
         free_blob(tasks[0]);
         free(tasks);
         close_service_connection(conn);
         *args->done = true;
         return;
      }
      int layer_id = tasks[0]->id;
      cnn_model *model = (cnn_model*)ctxt->model;
      layer *l = &model->net->layers[layer_id];
      blob *task;

      if (is_lip_layer(model, layer_id))
      {
         task = new_blob_and_alloc_data(layer_id, tasks[0]->size);
         memcpy(task->data, tasks[0]->data, tasks[0]->size);
      }
      else
      {
         for (int c = 1; c < ctxt->total_cli_num; c++){
            tasks[c] = recv_data(conn);
            total_recv += tasks[c]->size;
            if (tasks[c]->id != layer_id){
               printf("Inconsistent task data!\n");
               exit(1);
            }
         }

         bool can_reuse = can_reuse_lop_output(model, layer_id);

         // Reassemble data.
         task = new_blob_and_alloc_data(layer_id, l->inputs * sizeof(float));
         int num_parts = ctxt->total_cli_num;
         for (int c = 0; c < num_parts; c++){
            uint8_t *data_to_copy;
            if (can_reuse && c == ctxt->this_cli_id){
               layer *prev_l = &model->net->layers[layer_id - 1];
               data_to_copy = prev_l->output;
            } else {
               data_to_copy = tasks[c]->data;
            }

            int input_offset = sizeof(float) * get_lop_input_offset(l, c, num_parts);
            size_t input_size = get_lop_input_size(l, c, num_parts);
            memcpy(task->data + input_offset, data_to_copy, input_size);
         }
      }

      // Process data.
      blob *result = process_task_weightpart(ctxt, task);
      free_blob(task);
      free_blob(tasks[0]);
      if (!is_lip_layer(model, layer_id))
      {
         for (int c = 1; c < ctxt->total_cli_num; c++){
            free_blob(tasks[c]);
         }
      }
      free(tasks);
      /*enqueue(ctxt->result_queue_weightpart, result);
      free_blob(result);*/

      // Why not send it back directly?? Try that first...
      total_send += result->size;
      send_request("results_weight", 20, conn);
      send_data(result, conn);
      free_blob(result);

      // This is required because measurements become skewed if the os sometimes returns immediately to send.
      blob *confirm = recv_data(conn);
printf("done: %d\n", sys_now() - time_start);

printf("total recv: %d kB,  total send: %d kB\n", total_recv / 1000, total_send / 1000);
   }
}

void steal_partition_and_perform_inference_thread(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   /*Check gateway for possible stealing victims*/
   service_conn* conn;
   blob* temp;

   bool thread_running = false;
   bool thread_done = false;
   sys_thread_t t;

   steal_weightpart_args args = {0};
   args.ctxt = ctxt;
   args.done = &thread_done;

   while(1){
      conn = connect_service(TCP, ctxt->gateway_local_addr, WORK_STEAL_PORT);
      send_request("steal_gateway", 20, conn);
      temp = recv_data(conn);
      close_service_connection(conn);
      if(temp->id == -1){
         free_blob(temp);
         sys_sleep(100);
         continue;
      }

      if (!thread_running)
      {
         thread_running = true;
         t = sys_thread_new("steal_weightpart_thread", steal_weightpart_thread, &args, 0, 0);
      } else if (thread_done) {
         sys_thread_join(t);
         thread_running = false;
         thread_done = false;
      }

      conn = connect_service(TCP, (const char *)temp->data, WORK_STEAL_PORT);
      send_request("steal_client", 20, conn);
      free_blob(temp);
      temp = recv_data(conn);
      if(temp->id == -1){
         free_blob(temp);
         sys_sleep(100);
         continue;
      }
      bool data_ready = true;
#if DATA_REUSE
      blob* reuse_info_blob = recv_data(conn);
      bool* reuse_data_is_required = (bool*) reuse_info_blob->data;
      request_reuse_data(ctxt, temp, reuse_data_is_required);
      if(!need_reuse_data_from_gateway(reuse_data_is_required)) data_ready = false;
#if DEBUG_DEEP_EDGE
      printf("====================Processing task id is %d, data source is %d, frame_seq is %d====================\n", get_blob_task_id(temp), get_blob_cli_id(temp), get_blob_frame_seq(temp));
      printf("Request data from gateway, is the reuse data ready? ...\n");
      print_reuse_data_is_required(reuse_data_is_required);
#endif

      free_blob(reuse_info_blob);
#endif
      close_service_connection(conn);
      process_task(ctxt, temp, data_ready);
      free_blob(temp);
   }
}


/*Function handling steal reqeust*/
#if DATA_REUSE
void* steal_client_reuse_aware(void* srv_conn, void* arg){
   //printf("steal_client_reuse_aware ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* edge_model = (cnn_model*)(ctxt->model);

   blob* temp = try_dequeue(ctxt->task_queue);
   if(temp == NULL){
      char data[20]="empty";
      temp = new_blob_and_copy_data(-1, 20, (uint8_t*)data);
      send_data(temp, conn);
      free_blob(temp);
      return NULL;
   }
#if DEBUG_DEEP_EDGE
   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   processing_cli_id = get_client_id(ip_addr, ctxt);

   printf("Stolen local task is %d (task id: %d, from %d:%s)\n", temp->id, get_blob_task_id(temp), processing_cli_id, ip_addr);
#endif

   uint32_t task_id = get_blob_task_id(temp);
   bool* reuse_data_is_required = (bool*)malloc(sizeof(bool)*4);
   uint32_t position;
   for(position = 0; position < 4; position++){
      reuse_data_is_required[position] = false;
   }

   if(edge_model->ftp_para_reuse->schedule[get_blob_task_id(temp)] == 1 && is_reuse_ready(edge_model->ftp_para_reuse, get_blob_task_id(temp))) {
      uint32_t position;
      int32_t* adjacent_id = get_adjacent_task_id_list(edge_model, task_id);
      for(position = 0; position < 4; position++){
         if(adjacent_id[position]==-1) continue;
         reuse_data_is_required[position] = true;
      }
      free(adjacent_id);
      blob* shrinked_temp = new_blob_and_copy_data(get_blob_task_id(temp),
                       (edge_model->ftp_para_reuse->shrinked_input_size[get_blob_task_id(temp)]),
                       (uint8_t*)(edge_model->ftp_para_reuse->shrinked_input[get_blob_task_id(temp)]));
      copy_blob_meta(shrinked_temp, temp);
      free_blob(temp);
      temp = shrinked_temp;
   }
   send_data(temp, conn);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif
   free_blob(temp);

   /*Send bool variables for different positions*/
   temp = new_blob_and_copy_data(task_id,
                       sizeof(bool)*4,
                       (uint8_t*)(reuse_data_is_required));
   free(reuse_data_is_required);
   send_data(temp, conn);
   free_blob(temp);

   return NULL;
}

void* update_coverage(void* srv_conn, void* arg){
   printf("update_coverage ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* edge_model = (cnn_model*)(ctxt->model);

   blob* temp = recv_data(conn);
#if DEBUG_DEEP_EDGE
   printf("set coverage for task %d\n", get_blob_task_id(temp));
#endif
   set_coverage(edge_model->ftp_para_reuse, get_blob_task_id(temp));
   set_missing(edge_model->ftp_para_reuse, get_blob_task_id(temp));
   free_blob(temp);
   return NULL;
}
#endif

void deepthings_serve_stealing_thread(void *arg){
#if DATA_REUSE
   const char* request_types[]={"steal_client", "update_coverage"};
   void* (*handlers[])(void*, void*) = {steal_client_reuse_aware, update_coverage};
#else
   const char* request_types[]={"steal_client"};
   void* (*handlers[])(void*, void*) = {steal_client};
#endif
   int wst_service = service_init(WORK_STEAL_PORT, TCP);
#if DATA_REUSE
   start_service(wst_service, TCP, request_types, 2, handlers, arg);
#else
   start_service(wst_service, TCP, request_types, 1, handlers, arg);
#endif
   close_service(wst_service);
}


void deepthings_stealer_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t cli_num, const char** edge_addr_list){


   device_ctxt* ctxt = deepthings_edge_init(N, M, fused_layers, network, weights, edge_id, cli_num, edge_addr_list);
   exec_barrier(START_CTRL, TCP, ctxt);

   sys_thread_t t1 = sys_thread_new("steal_partition_and_perform_inference_thread", steal_partition_and_perform_inference_thread, ctxt, 0, 0);
   sys_thread_t t2 = sys_thread_new("send_result_thread", send_result_thread, ctxt, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);

}

void gather_result(device_ctxt *ctxt, blob *result){
   int32_t cli_id = get_blob_cli_id(result);
   if (cli_id != get_this_client_id(ctxt)){
      printf("ERROR: Got result that was intended for other cli\n");
      return;
   }
   int32_t frame_seq = get_blob_frame_seq(result);
   int32_t task_id = get_blob_task_id(result);

   uint32_t num_nodes = enqueue(ctxt->results_pool[frame_seq], result);

   printf("gathering result frame: %d, task: %d   complete: %d/%d\n", frame_seq, task_id, num_nodes, ctxt->batch_size);

   if (num_nodes == ctxt->batch_size){
      printf("Results ready!\n");
      blob *ready = new_empty_blob(cli_id);
      annotate_blob(ready, cli_id, frame_seq, task_id);
      enqueue(ctxt->ready_pool, ready);
      free_blob(ready);
   }
}

/*Same implementation with result_gateway,just insert more profiling information*/
void* deepthings_result_back(void* srv_conn, void* arg){
   printf("result_back ... ... \n");
   device_ctxt* ctxt = (device_ctxt*)arg;
   service_conn *conn = (service_conn *)srv_conn;
#if DEBUG_FLAG
   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   processing_cli_id = get_client_id(ip_addr, ctxt);
#if DEBUG_TIMING
   double total_time;
   uint32_t total_frames;
   double now;
   uint32_t i;
#endif
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");
#endif
   blob* temp = recv_data(conn);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif
/*#if DEBUG_FLAG
   printf("Result from %d: %s total number recved is %d\n", processing_cli_id, ip_addr, ctxt->results_counter);
#endif*/
   gather_result(ctxt, temp);
   free_blob(temp);

   return NULL;
}
void deepthings_collect_result_thread(void *arg){
   const char* request_types[]={"result_back"};
   void* (*handlers[])(void*, void*) = {deepthings_result_back};
   int result_service = service_init(RESULT_COLLECT_PORT, TCP);
   start_service(result_service, TCP, request_types, 1, handlers, arg);
   close_service(result_service);
}

void gather_local_results_thread(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   blob *temp;

   while(1){
      temp = dequeue(ctxt->result_queue);
      gather_result(ctxt, temp);
      free_blob(temp);
   }
}

int on_weight_part_push(void *svc_conn, void *arg){
   service_conn *conn = (service_conn*)svc_conn;
   device_ctxt *ctxt = (device_ctxt*)arg;

   // Get other client id.
   char ip_addr[ADDRSTRLEN];
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   int32_t cli_id = get_client_id(ip_addr, ctxt);

static int total_send = 0;
      blob *task = dequeue(ctxt->task_queue_weightpart[cli_id]);
      if (task->id == -1)
      {
         // Stop the connection.
         send_data(task, conn);
         free_blob(task);
         return 1;
      }
      total_send += task->size;
   send_data(task, conn);
   int layer_id = task->id;
   if (!is_lip_layer(ctxt->model, layer_id))
   {
      for (int c = 1; c < ctxt->total_cli_num; c++)
      {
         task = dequeue(ctxt->task_queue_weightpart[cli_id]);
         if (task->id != layer_id)
         {
            printf("Inconsistent task layer ids\n");
            exit(1);
         }
         total_send += task->size;
      send_data(task, conn);
   }
   }
printf("total send: %d kB\n", total_send / 1000);

   return 0;
}

void store_weight_part_result(device_ctxt *ctxt, blob *result)
{
   int32_t cli_id = get_blob_cli_id(result);
   int32_t frame_seq = get_blob_frame_seq(result);
   int32_t task_id = get_blob_task_id(result);

   uint32_t num_nodes = enqueue(ctxt->results_pool_weightpart, result);

   //printf("gathering result weightpart cli: %d, frame: %d, task: %d   complete: %d/%d\n", cli_id, frame_seq, task_id, num_nodes, ctxt->total_cli_num);

   if (num_nodes == ctxt->total_cli_num){
      //printf("Weight results ready!\n");
      blob *ready = new_empty_blob(cli_id);
      annotate_blob(ready, cli_id, frame_seq, task_id);
      enqueue(ctxt->ready_pool_weightpart, ready);
      free_blob(ready);
   }
}

void *on_weight_part_results(void *svc_conn, void *arg){
   service_conn *conn = (service_conn*)svc_conn;
   device_ctxt *ctxt = (device_ctxt*)arg;

   blob *result = recv_data(conn);
   static int total_recv = 0;
   total_recv += result->size;
printf("total recv: %d kB\n", total_recv / 1000);
   store_weight_part_result(ctxt, result);
   free_blob(result);

   blob *confirm = new_empty_blob(0);
   send_data(confirm, conn);
}

void weight_part_service_thread(void *arg){
   device_ctxt *ctxt = (device_ctxt*)arg;

   const char *request_types[] = { "results_weight" };
   void *(*handlers[])(void*, void*) = { on_weight_part_results };

   int push_service = service_init(WEIGHT_PART_PORT, TCP);
   start_parallel_push_service(push_service, TCP, on_weight_part_push, request_types, 1, handlers, arg);
   close_service(push_service);
}


void deepthings_victim_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t cli_num, const char** edge_addr_list){


   device_ctxt* ctxt = deepthings_edge_init(N, M, fused_layers, network, weights, edge_id, cli_num, edge_addr_list);
   exec_barrier(START_CTRL, TCP, ctxt);

   sys_thread_t t1 = sys_thread_new("partition_frame_and_perform_inference_thread", partition_frame_and_perform_inference_thread, ctxt, 0, 0);
   sys_thread_t t2 = sys_thread_new("gather_local_results_thread", gather_local_results_thread, ctxt, 0, 0);
   sys_thread_t t3 = sys_thread_new("deepthings_collect_result_thread", deepthings_collect_result_thread, ctxt, 0, 0);
   sys_thread_t t5 = sys_thread_new("deepthings_serve_stealing_thread", deepthings_serve_stealing_thread, ctxt, 0, 0);
   sys_thread_t t6 = sys_thread_new("weight_part_service_thread", weight_part_service_thread, ctxt, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);
   sys_thread_join(t5);
   sys_thread_join(t6);

#ifdef NNPACK
   pthreadpool_destroy(((cnn_model*)ctxt->model)->net->threadpool);
   nnp_deinitialize();
#endif
}
