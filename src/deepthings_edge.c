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

device_ctxt* deepthings_edge_init(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t cli_num, const char** edge_addr_list){
   device_ctxt* ctxt = init_client(edge_id, FRAME_NUM);
   cnn_model* model = load_cnn_model(network, weights);
   model->ftp_para = preform_ftp(N, M, fused_layers, model->net_para);
#if DATA_REUSE
   model->ftp_para_reuse = preform_ftp_reuse(model->net_para, model->ftp_para);
#endif
   ctxt->model = model;
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


void partition_frame_and_perform_inference_thread(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* model = (cnn_model*)(ctxt->model);
   blob* temp;
   uint32_t frame_num;
   bool* reuse_data_is_required;
   for(frame_num = 0; frame_num < FRAME_NUM; frame_num ++){
      /*Wait for i/o device input*/
      /*recv_img()*/

      /*Load image and partition, fill task queues*/
      load_image_as_model_input(model, frame_num);
      partition_and_enqueue(ctxt, frame_num);
      register_client(ctxt);

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

      /*Unregister and prepare for next image*/
      cancel_client(ctxt);
   }
}




void steal_partition_and_perform_inference_thread(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   /*Check gateway for possible stealing victims*/
   service_conn* conn;
   blob* temp;
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


/*defined in gateway.h from darkiot
void send_result_thread;
*/


/*Function handling steal reqeust*/
#if DATA_REUSE
void* steal_client_reuse_aware(void* srv_conn, void* arg){
   printf("steal_client_reuse_aware ... ... \n");
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

   printf("gathering result frame: %d, task: %d   complete: %d/%d\n", frame_seq, task_id, ctxt->results_pool[frame_seq]->number_of_node+1, ctxt->batch_size);

   enqueue(ctxt->results_pool[frame_seq], result);
   if (ctxt->results_pool[frame_seq]->number_of_node == ctxt->batch_size){
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

void deepthings_merge_result_thread(void *arg){
   cnn_model* model = (cnn_model*)(((device_ctxt*)(arg))->model);
   blob* temp;
   int32_t cli_id;
   int32_t frame_seq;
   while(1){
      temp = dequeue_and_merge((device_ctxt*)arg);
      cli_id = get_blob_cli_id(temp);
      frame_seq = get_blob_frame_seq(temp);
#if DEBUG_FLAG
      printf("Client %d, frame sequence number %d, all partitions are merged in deepthings_merge_result_thread\n", cli_id, frame_seq);
#endif
      float* fused_output = (float*)(temp->data);
      image_holder img = load_image_as_model_input(model, get_blob_frame_seq(temp));
      set_model_input(model, fused_output);
      forward_all(model, model->ftp_para->fused_layers);
      draw_object_boxes(model, get_blob_frame_seq(temp));
      free_image_holder(model, img);
      free_blob(temp);
#if DEBUG_FLAG
      printf("Client %d, frame sequence number %d, finish processing\n", cli_id, frame_seq);
#endif
   }
}

void gather_local_results(void *arg){
   device_ctxt* ctxt = (device_ctxt*)arg;
   blob *temp;

   while(1){
      temp = dequeue(ctxt->result_queue);
      gather_result(ctxt, temp);
      free_blob(temp);
   }
}

void deepthings_victim_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t cli_num, const char** edge_addr_list){


   device_ctxt* ctxt = deepthings_edge_init(N, M, fused_layers, network, weights, edge_id, cli_num, edge_addr_list);
   exec_barrier(START_CTRL, TCP, ctxt);

   sys_thread_t t1 = sys_thread_new("partition_frame_and_perform_inference_thread", partition_frame_and_perform_inference_thread, ctxt, 0, 0);
   sys_thread_t t2 = sys_thread_new("gather_local_results", gather_local_results, ctxt, 0, 0);
   sys_thread_t t3 = sys_thread_new("deepthings_collect_result_thread", deepthings_collect_result_thread, ctxt, 0, 0);
   sys_thread_t t4 = sys_thread_new("deepthings_merge_result_thread", deepthings_merge_result_thread, ctxt, 0, 0);
   sys_thread_t t5 = sys_thread_new("deepthings_serve_stealing_thread", deepthings_serve_stealing_thread, ctxt, 0, 0);

   sys_thread_join(t1);
   sys_thread_join(t2);
   sys_thread_join(t3);
   sys_thread_join(t4);
   sys_thread_join(t5);

#ifdef NNPACK
   pthreadpool_destroy(((cnn_model*)ctxt->model)->net->threadpool);
   nnp_deinitialize();
#endif
}
