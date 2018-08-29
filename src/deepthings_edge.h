#ifndef DEEPTHINGS_EDGE_H
#define DEEPTHINGS_EDGE_H
#include "darkiot.h"
#include "configure.h"

void deepthings_stealer_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t cli_num, const char** edge_addr_list);
void deepthings_victim_edge(uint32_t N, uint32_t M, uint32_t fused_layers, char* network, char* weights, uint32_t edge_id, uint32_t cli_num, const char** edge_addr_list);

#endif
