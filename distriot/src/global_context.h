#ifndef GLOBAL_CONTEXT_H
#define GLOBAL_CONTEXT_H
#include "thread_safe_queue.h"
#include <string.h>
#define ADDR_LEN 64
#define MAX_QUEUE_SIZE 256

typedef struct dev_ctxt {
   // Results that have been gathered from own node or other stealer nodes. Indexed by frame_seq.
   thread_safe_queue** results_pool;
   // Signals that enough results have been gathered to finalize inference.
   thread_safe_queue* ready_pool;
   // Holds nodes that are registered for work stealing.
   thread_safe_queue* registration_list;
   char** addr_list;
   uint32_t total_cli_num;

   thread_safe_queue* task_queue;
   // Results that the own node generated.
   thread_safe_queue* result_queue;
   uint32_t this_cli_id;

   // Weight partitioning queues.
   thread_safe_queue *results_pool_weightpart;
   thread_safe_queue *ready_pool_weightpart;
   thread_safe_queue **task_queue_weightpart;
   thread_safe_queue *result_queue_weightpart;

   uint32_t batch_size;/*Number of tasks to merge*/
   void *model;/*pointers to execution model*/
   uint32_t total_frames;/*max number of input frames*/

   char gateway_local_addr[ADDR_LEN];
   char gateway_public_addr[ADDR_LEN];

} device_ctxt;

device_ctxt* init_context(uint32_t cli_id, uint32_t cli_num, const char** edge_addr_list);
void set_batch_size(device_ctxt* ctxt, uint32_t size);
void set_gateway_local_addr(device_ctxt* ctxt, const char* addr);
void set_gateway_public_addr(device_ctxt* ctxt, const char* addr);
void set_total_frames(device_ctxt* ctxt, uint32_t frame_num);
#endif
