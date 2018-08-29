#include "gateway.h"

/*Allocated spaces for gateway devices*/
device_ctxt* init_gateway(uint32_t cli_num, const char** edge_addr_list){

   device_ctxt* ctxt = (device_ctxt*)malloc(sizeof(device_ctxt));
   uint32_t i;

   // Work stealing.
   ctxt->registration_list = new_queue(MAX_QUEUE_SIZE);
   ctxt->total_cli_num = cli_num;
   ctxt->addr_list = (char**)malloc(sizeof(char*)*cli_num);
   for(i = 0; i < cli_num; i++){
      ctxt->addr_list[i] = (char*)malloc(sizeof(char)*ADDR_LEN);
      strcpy(ctxt->addr_list[i], edge_addr_list[i]);
   }

   return ctxt;
}

void* register_gateway(void* srv_conn, void *arg){
   printf("register_gateway ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   char ip_addr[ADDRSTRLEN];
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   blob* temp = new_blob_and_copy_data(get_client_id(ip_addr, ctxt), ADDRSTRLEN, (uint8_t*)ip_addr);
   enqueue(ctxt->registration_list, temp);
   free_blob(temp);
#if DEBUG_FLAG
   queue_node* cur = ctxt->registration_list->head;
   if (ctxt->registration_list->head == NULL){
      printf("No client is registered!\n");
   }
   while (1) {
      if (cur->next == NULL){
         printf("%d: %s,\n", cur->item->id, ((char*)(cur->item->data)));
         break;
      }
      printf("%d: %s\n", cur->item->id, ((char*)(cur->item->data)));
      cur = cur->next;
   }
#endif
   return NULL;
}

void* cancel_gateway(void* srv_conn, void *arg){
   printf("cancel_gateway ... ... \n");
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   char ip_addr[ADDRSTRLEN];
   inet_ntop(conn->serv_addr_ptr->sin_family, &(conn->serv_addr_ptr->sin_addr), ip_addr, ADDRSTRLEN);
   int32_t cli_id = get_client_id(ip_addr, ctxt);
   remove_by_id(ctxt->registration_list, cli_id);
   return NULL;
}

void* steal_gateway(void* srv_conn, void *arg){
   service_conn *conn = (service_conn *)srv_conn;
   device_ctxt* ctxt = (device_ctxt*)arg;
   blob* temp = try_dequeue(ctxt->registration_list);
   if(temp == NULL){
      char ip_addr[ADDRSTRLEN]="empty";
      temp = new_blob_and_copy_data(-1, ADDRSTRLEN, (uint8_t*)ip_addr);
   }else{
      enqueue(ctxt->registration_list, temp);
   }
   send_data(temp, conn);
   free_blob(temp);
   return NULL;
}

void work_stealing_thread(void *arg){
   const char* request_types[]={"register_gateway", "cancel_gateway", "steal_gateway"};
   void* (*handlers[])(void*,void*) = {register_gateway, cancel_gateway, steal_gateway};
   int wst_service = service_init(WORK_STEAL_PORT, TCP);
   start_service(wst_service, TCP, request_types, 3, handlers, arg);
   close_service(wst_service);
}
