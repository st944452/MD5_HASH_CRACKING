#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <openssl/md5.h>

#define MAX_PASSWORD_LENGTH 32
#define MAX_HASH_LENGTH 16
#define BATCH_SIZE 1024
#define THREADS_PER_BLOCK 256

void hex_to_bytes(const char* hex, unsigned char* bytes) {
    for (int i = 0; i < 16; i++) {
        sscanf(hex + 2 * i, "%2hhx", &bytes[i]);
    }
}

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { fprintf(stderr, "%s failed with error code %d\n", msg, err); exit(1); }

const char* kernelSource = "__kernel void md5_crack_kernel(                \n" \
"   __global const unsigned char* hashes,                                \n" \
"   __global const unsigned char* target_hash,                           \n" \
"   __global int* match_index,                                           \n" \
"   const int num_passwords) {                                           \n" \
"   int idx = get_global_id(0);                                          \n" \
"   if (idx >= num_passwords) return;                                    \n" \
"                                                                        \n" \
"   bool match = true;                                                   \n" \
"   for (int i = 0; i < 16; i++) {                                       \n" \
"       if (hashes[idx * 16 + i] != target_hash[i]) {                    \n" \
"           match = false;                                               \n" \
"           break;                                                       \n" \
"       }                                                                \n" \
"   }                                                                    \n" \
"   if (match) {                                                         \n" \
"       *match_index = idx;                                              \n" \
"   }                                                                    \n" \
"}\n";

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <target_hash> <dictionary_file>\n", argv[0]);
        return 1;
    }

    const char* target_hash_hex = argv[1];
    const char* dictionary_file = argv[2];

    unsigned char target_hash[16];
    hex_to_bytes(target_hash_hex, target_hash);

    FILE* file = fopen(dictionary_file, "r");
    if (!file) {
        perror("Failed to open dictionary file");
        return 1;
    }

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, NULL); CHECK_ERROR(ret, "clGetPlatformIDs");
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL); CHECK_ERROR(ret, "clGetDeviceIDs");
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret); CHECK_ERROR(ret, "clCreateContext");
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret); CHECK_ERROR(ret, "clCreateCommandQueue");

    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &ret); CHECK_ERROR(ret, "clCreateProgramWithSource");
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Kernel build error:\n%s\n", log);
        free(log);
        exit(1);
    }
    kernel = clCreateKernel(program, "md5_crack_kernel", &ret); CHECK_ERROR(ret, "clCreateKernel");

    unsigned char hashes[BATCH_SIZE * MAX_HASH_LENGTH];
    char passwords[BATCH_SIZE][MAX_PASSWORD_LENGTH + 1];
    int match_index = -1;

    cl_mem device_hashes = clCreateBuffer(context, CL_MEM_READ_ONLY, BATCH_SIZE * MAX_HASH_LENGTH, NULL, &ret); CHECK_ERROR(ret, "clCreateBuffer device_hashes");
    cl_mem device_target_hash = clCreateBuffer(context, CL_MEM_READ_ONLY, 16 * sizeof(unsigned char), NULL, &ret); CHECK_ERROR(ret, "clCreateBuffer device_target_hash");
    cl_mem device_match_index = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &ret); CHECK_ERROR(ret, "clCreateBuffer device_match_index");

    ret = clEnqueueWriteBuffer(command_queue, device_target_hash, CL_TRUE, 0, 16 * sizeof(unsigned char), target_hash, 0, NULL, NULL); CHECK_ERROR(ret, "clEnqueueWriteBuffer target_hash");

    // Get max work group size for the device
    size_t max_work_group_size;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);

    size_t global_item_size;
    size_t local_item_size = THREADS_PER_BLOCK;
    if (local_item_size > max_work_group_size) {
        local_item_size = max_work_group_size;
    }

    int batch_size = 0;

    while (fgets(passwords[batch_size], MAX_PASSWORD_LENGTH + 1, file) != NULL) {
        passwords[batch_size][strcspn(passwords[batch_size], "\n")] = 0;

        unsigned char hash[16];
        MD5((unsigned char*)passwords[batch_size], strlen(passwords[batch_size]), hash);
        memcpy(&hashes[batch_size * MAX_HASH_LENGTH], hash, MAX_HASH_LENGTH);

        batch_size++;

        if (batch_size == BATCH_SIZE) {
            global_item_size = ((batch_size + local_item_size - 1) / local_item_size) * local_item_size;

            ret = clEnqueueWriteBuffer(command_queue, device_hashes, CL_TRUE, 0, batch_size * MAX_HASH_LENGTH, hashes, 0, NULL, NULL); CHECK_ERROR(ret, "clEnqueueWriteBuffer device_hashes");

            ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&device_hashes);
            ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&device_target_hash);
            ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&device_match_index);
            ret = clSetKernelArg(kernel, 3, sizeof(int), &batch_size);

            ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL); CHECK_ERROR(ret, "clEnqueueNDRangeKernel");
            clFinish(command_queue);

            ret = clEnqueueReadBuffer(command_queue, device_match_index, CL_TRUE, 0, sizeof(int), &match_index, 0, NULL, NULL); CHECK_ERROR(ret, "clEnqueueReadBuffer device_match_index");

            if (match_index >= 0 && match_index < batch_size) {
                printf("Password found: %s\n", passwords[match_index]);
                break;
            }

            batch_size = 0;
            match_index = -1;
            ret = clEnqueueWriteBuffer(command_queue, device_match_index, CL_TRUE, 0, sizeof(int), &match_index, 0, NULL, NULL);
        }
    }

    if (match_index == -1 && batch_size > 0) {
        global_item_size = ((batch_size + local_item_size - 1) / local_item_size) * local_item_size;

        ret = clEnqueueWriteBuffer(command_queue, device_hashes, CL_TRUE, 0, batch_size * MAX_HASH_LENGTH, hashes, 0, NULL, NULL); CHECK_ERROR(ret, "clEnqueueWriteBuffer device_hashes");

        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&device_hashes);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&device_target_hash);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&device_match_index);
        ret = clSetKernelArg(kernel, 3, sizeof(int), &batch_size);

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL); CHECK_ERROR(ret, "clEnqueueNDRangeKernel");
        clFinish(command_queue);

        ret = clEnqueueReadBuffer(command_queue, device_match_index, CL_TRUE, 0, sizeof(int), &match_index, 0, NULL, NULL); CHECK_ERROR(ret, "clEnqueueReadBuffer device_match_index");

        if (match_index >= 0 && match_index < batch_size) {
            printf("Password found: %s\n", passwords[match_index]);
        } else {
            printf("Password not found in the dictionary.\n");
        }
    }

    fclose(file);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(device_hashes);
    clReleaseMemObject(device_target_hash);
    clReleaseMemObject(device_match_index);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
