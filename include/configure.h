#ifndef CONFIGURE_H
#define CONFIGURE_H

/*Partitioning paramters*/
#define FUSED_LAYERS_MAX 16
#define PARTITIONS_W_MAX 6
#define PARTITIONS_H_MAX 6
#define PARTITIONS_MAX 36
#define THREAD_NUM 1
#define DATA_REUSE 1

/*Debugging information for different components*/
#define DEBUG_INFERENCE 1
#define DEBUG_FTP 0
#define DEBUG_SERIALIZATION 0
#define DEBUG_DEEP_GATEWAY 1
#define DEBUG_DEEP_EDGE 1

/*Print timing and communication size information*/
#define DEBUG_TIMING 1
#define DEBUG_COMMU_SIZE 1

/*Configuration parameters for DistrIoT*/
#define GATEWAY_PUBLIC_ADDR "192.168.0.2"
#define GATEWAY_LOCAL_ADDR "192.168.0.2"
#define EDGE_ADDR_LIST    { \
  "192.168.0.3", \
  "192.168.0.4", \
  "192.168.0.5", \
  "192.168.0.6", \
  "192.168.0.7", \
  "192.168.0.8"}
#ifndef MAX_EDGE_NUM
#define MAX_EDGE_NUM 6
#pragma message "Using default edge dev count of 6"
#else
#define STRING2(x) #x
#define STRING(x) STRING2(x)
#pragma message(STRING(MAX_EDGE_NUM) " edge devices")
#endif
#define FRAME_NUM 1

#endif
