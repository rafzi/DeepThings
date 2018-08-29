#ifndef GATEWAY_H
#define GATEWAY_H
#include "darkiot.h"

device_ctxt* init_gateway(uint32_t cli_num, const char** edge_addr_list);
// Services result commands and collects the results. Reports ready if all collected.
void collect_result_thread(void *arg);
// If ready (from result_thread) discards results??
void merge_result_thread(void *arg);
// Services register, cancel and steal commands.
void work_stealing_thread(void *arg);
// Called when a client registers at the gateway. Save clients address for stealing requests.
void* register_gateway(void* srv_conn, void *arg);
// Called when a registration is canceled. Remove clients address.
void* cancel_gateway(void* srv_conn, void *arg);
// Called when a client requested to steal work. If work available return clients address.
void* steal_gateway(void* srv_conn, void *arg);

#endif
