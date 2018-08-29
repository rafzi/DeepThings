#ifndef CLIENT_H
#define CLIENT_H
#include "darkiot.h"

device_ctxt* init_client(uint32_t cli_id, int frame_num);
// Keeps trying to steal task and processes it in this thread.
void steal_and_process_thread(void *arg);
// Generates tasks and processes them in this thread. While processing, be registered at gateway.
void generate_and_process_thread(void *arg);
// Waits for results and sends them to the gateway.
void send_result_thread(void *arg);
// Services steal requests from other clients.
void serve_stealing_thread(void *arg);
// Registers the client as available for stealing at the gateway.
void register_client(device_ctxt* ctxt);
// Cancels being available for stealing.
void cancel_client(device_ctxt* ctxt);
// Respond to a steal request by sending task if available.
void* steal_client(void* srv_conn, void *arg);

#endif
