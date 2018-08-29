#include "deepthings_gateway.h"
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

device_ctxt* deepthings_gateway_init(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t total_edge_number, const char** addr_list){
   device_ctxt* ctxt = init_gateway(total_edge_number, addr_list);
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

   return ctxt;
}


#if DATA_REUSE
void notify_coverage(device_ctxt* ctxt, blob* task_input_blob, uint32_t cli_id){
   char data[20]="empty";
   blob* temp = new_blob_and_copy_data(cli_id, 20, (uint8_t*)data);
   copy_blob_meta(temp, task_input_blob);
   service_conn* conn;
   conn = connect_service(TCP, get_client_addr(cli_id, ctxt), WORK_STEAL_PORT);
   send_request("update_coverage", 20, conn);
   send_data(temp, conn);
   free_blob(temp);
   close_service_connection(conn);
}
#endif



#if DATA_REUSE
static overlapped_tile_data* overlapped_data_pool[MAX_EDGE_NUM][PARTITIONS_MAX];
/*
static bool partition_coverage[MAX_EDGE_NUM][PARTITIONS_MAX];
*/
void* recv_reuse_data_from_edge(void* srv_conn, void* arg){
   printf("collecting_reuse_data ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   cnn_model* gateway_model = (cnn_model*)(((device_ctxt*)(arg))->model);

   int32_t cli_id;
   int32_t task_id;

   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   processing_cli_id = get_client_id(ip_addr, (device_ctxt*)arg);
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");

   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
   task_id = get_blob_task_id(temp);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif

#if DEBUG_DEEP_GATEWAY
   printf("Overlapped data for client %d, task %d is collected from %d: %s, size is %d\n", cli_id, task_id, processing_cli_id, ip_addr, temp->size);
#endif
   if(overlapped_data_pool[cli_id][task_id] != NULL)
      free_self_overlapped_tile_data(gateway_model,  overlapped_data_pool[cli_id][task_id]);
   overlapped_data_pool[cli_id][task_id] = self_reuse_data_deserialization(gateway_model, task_id, (float*)temp->data, get_blob_frame_seq(temp));

   if(processing_cli_id != cli_id) notify_coverage((device_ctxt*)arg, temp, cli_id);
   free_blob(temp);

   return NULL;
}

void* send_reuse_data_to_edge(void* srv_conn, void* arg){
   printf("handing_out_reuse_data ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   cnn_model* gateway_model = (cnn_model*)(ctxt->model);

   int32_t cli_id;
   int32_t task_id;
   uint32_t frame_num;
   blob* temp = recv_data(conn);
   cli_id = get_blob_cli_id(temp);
   task_id = get_blob_task_id(temp);
   frame_num = get_blob_frame_seq(temp);
   free_blob(temp);

#if DEBUG_DEEP_GATEWAY
   char ip_addr[ADDRSTRLEN];
   int32_t processing_cli_id;
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   processing_cli_id = get_client_id(ip_addr, ctxt);
   if(processing_cli_id < 0)
      printf("Client IP address unknown ... ...\n");
#endif

   blob* reuse_info_blob = recv_data(conn);
   bool* reuse_data_is_required = (bool*)(reuse_info_blob->data);

#if DEBUG_DEEP_GATEWAY
   printf("Overlapped data for client %d, task %d is required by %d: %s is \n", cli_id, task_id, processing_cli_id, ip_addr);
   print_reuse_data_is_required(reuse_data_is_required);
#endif
   uint32_t position;
   int32_t* adjacent_id = get_adjacent_task_id_list(gateway_model, task_id);

   for(position = 0; position < 4; position++){
      if(adjacent_id[position]==-1) continue;
      if(reuse_data_is_required[position]){
#if DEBUG_DEEP_GATEWAY
         printf("place_self_deserialized_data for client %d, task %d, the adjacent task is %d\n", cli_id, task_id, adjacent_id[position]);
#endif
         place_self_deserialized_data(gateway_model, adjacent_id[position], overlapped_data_pool[cli_id][adjacent_id[position]]);
      }
   }
   free(adjacent_id);
   temp = adjacent_reuse_data_serialization(ctxt, task_id, frame_num, reuse_data_is_required);
   free_blob(reuse_info_blob);
   send_data(temp, conn);
#if DEBUG_COMMU_SIZE
   commu_size = commu_size + temp->size;
#endif
   free_blob(temp);

   return NULL;
}

#endif

void deepthings_work_stealing_thread(void *arg){
#if DATA_REUSE
   const char* request_types[]={"register_gateway", "cancel_gateway", "steal_gateway", "reuse_data", "request_reuse_data"};
   void* (*handlers[])(void*, void*) = {register_gateway, cancel_gateway, steal_gateway, recv_reuse_data_from_edge, send_reuse_data_to_edge};
#else
   const char* request_types[]={"register_gateway", "cancel_gateway", "steal_gateway"};
   void* (*handlers[])(void*, void*) = {register_gateway, cancel_gateway, steal_gateway};
#endif

   int wst_service = service_init(WORK_STEAL_PORT, TCP);
#if DATA_REUSE
   start_service(wst_service, TCP, request_types, 5, handlers, arg);
#else
   start_service(wst_service, TCP, request_types, 3, handlers, arg);
#endif
   close_service(wst_service);
}


void deepthings_gateway(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t total_edge_number, const char** addr_list){
   device_ctxt* ctxt = deepthings_gateway_init(N, M, fused_layers, network, weights, total_edge_number, addr_list);
   sys_thread_t t1 = sys_thread_new("deepthings_work_stealing_thread", deepthings_work_stealing_thread, ctxt, 0, 0);
   exec_barrier(START_CTRL, TCP, ctxt);
#if DEBUG_TIMING
   start_time = sys_now_in_sec();
#endif
   sys_thread_join(t1);
}
