// System includes
#include <assert.h>
#include <ctime>
#include <math.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
// #include <helper_functions.h>
// #include <helper_cuda.h>

#include <device_launch_parameters.h>

#define MAX_4CHAR_VAL 1679616
#define T36_POWER_RANGE 8
#define MAC_POS_SIZE 2821109907456

#define print_plaintext(x) \
    for (int i = 0; i < 4; i++) { \
        printf("%c", (char)((x) >> (24 - i * 8))); \
    }

#define print_hex(x) \
    for (int i = 0; i < 4; i++) { \
        printf("%02x", (u8)((x) >> (24 - i * 8))); \
    }
    

u8 T_CHARSET[36] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
};

__host__ u64* calculateCMACRange() {
	u64* range;
	gpuErrorCheck(cudaMallocManaged(&range, 1 * sizeof(u64)));
	int threadCount = BLOCKS * THREADS;
	double keyRange = pow(36, T36_POWER_RANGE);
	double threadRange = keyRange / threadCount;
	*range = ceil(threadRange);

    printf("Blocks                        : %d\n", BLOCKS);
	printf("Threads                       : %d\n", THREADS);
	printf("Total Thread count            : %d\n", threadCount);
	printf("Key Range (36 power)          : %d\n", T36_POWER_RANGE);
	printf("Key Range (decimal)           : %.0f\n", keyRange);
	printf("Each Thread Key Range         : %.2f\n", threadRange);
	printf("Each Thread Key Range (kernel): %llu\n", range[0]);
	printf("Total encryptions             : %.0f\n", ceil(threadRange) * threadCount);
	printf("-------------------------------\n");
	
	return range;
}

__device__ inline u32 derive_plaintext(u64 index, u8* charset)
{
    u32 res = 0;
    for (int i = 0; i < 4; i++) {
        res = res * 256 + (u32)charset[index % 36];
        index /= 36;
    }
    return res;
}



__global__ void blue_mesh_cmac_exhaustive_search(u32* pt, u32* ct, u32* rk, u32* rk1, u32* t0G, u32* t4G, u32* rconG, u64* range, u8* SAES, u8* CHARSET)
{
    u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex >= MAC_POS_SIZE) {
        return;
    }

    int warpThreadIndex = threadIdx.x & 31;
    //	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;
    // <SHARED MEMORY>
    __shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
    __shared__ u32 rconS[RCON_SIZE];
    __shared__ u32 ctS[U32_SIZE];
    //	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
    __shared__ u8 Sbox[64][32][4];
    __shared__ u8 t_charset[36];

    if (threadIdx.x < TABLE_SIZE) {
        for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
            t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
            Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x];
        }
        //		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];		}
        //		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) { Sbox[threadIdx.x/4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x]; }
        if (threadIdx.x < U32_SIZE) {
            ctS[threadIdx.x] = ct[threadIdx.x];
        }
        if (threadIdx.x < RCON_SIZE) {
            rconS[threadIdx.x] = rconG[threadIdx.x];
        }
        if (threadIdx.x < 36) {
            t_charset[threadIdx.x] = CHARSET[threadIdx.x];
        }
    } // </SHARED MEMORY>
    __syncthreads(); // Wait until every thread is ready
    u32 rk0Init, rk1Init, rk2Init, rk3Init;
    rk0Init = rk[0];
    rk1Init = rk[1];
    rk2Init = rk[2];
    rk3Init = rk[3];

    u32 rk0Init1, rk1Init1, rk2Init1, rk3Init1;
    rk0Init1 = rk1[0];
    rk1Init1 = rk1[1];
    rk2Init1 = rk1[2];
    rk3Init1 = rk1[3];

    u32 pt0Init, pt1Init, pt2Init, pt3Init;
    pt0Init = pt[0];
    pt1Init = pt[1];
    pt2Init = pt[2];
    pt3Init = pt[3];

    u64 threadRange = *range;
    u64 threadRangeStart = threadIndex * threadRange;

    u64 index1, index2;
    index1 = threadRangeStart / (u64)MAX_4CHAR_VAL;
    index2 = threadRangeStart % (u64)MAX_4CHAR_VAL;

    u32 pt4Init, pt5Init, pt6Init, pt7Init;
    pt4Init = derive_plaintext(index1, t_charset);
    pt5Init = derive_plaintext(index2, t_charset);
    pt6Init = 0;
    pt7Init = 0;

    for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
        ///////////////////////////////////////////////////////////////////
        //              Calculate the 1st message block                  //
        ///////////////////////////////////////////////////////////////////
        u32 rk0, rk1, rk2, rk3;
        rk0 = rk0Init;
        rk1 = rk1Init;
        rk2 = rk2Init;
        rk3 = rk3Init;

        // Create plaintext as 32 bit unsigned integers
        u32 s0, s1, s2, s3;
        s0 = pt0Init;
        s1 = pt1Init;
        s2 = pt2Init;
        s3 = pt3Init;

        // First round just XORs input with key.
        s0 = s0 ^ rk0;
        s1 = s1 ^ rk1;
        s2 = s2 ^ rk2;
        s3 = s3 ^ rk3;
        u32 t0, t1, t2, t3;
        for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
            // Calculate round key
            u32 temp = rk3;
            rk0 = rk0 ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^ rconS[roundCount];
            rk1 = rk1 ^ rk0;
            rk2 = rk2 ^ rk1;
            rk3 = rk2 ^ rk3;
            // Table based round function
            t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk0;
            t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk1;
            t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk2;
            t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk3;
            s0 = t0;
            s1 = t1;
            s2 = t2;
            s3 = t3;
        }
        // Calculate the last round key
        u32 temp = rk3;
        rk0 = rk0 ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^ rconS[ROUND_COUNT_MIN_1];
        // Last round uses s-box directly and XORs to produce output.
        s0 = arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rk0;
        rk1 = rk1 ^ rk0;
        s1 = arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rk1;
        rk2 = rk2 ^ rk1;
        s2 = arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rk2;
        rk3 = rk2 ^ rk3;
        s3 = arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rk3;


        ///////////////////////////////////////////////////////////////////
        //              Calculate the 2nd message block                  //
        ///////////////////////////////////////////////////////////////////
        rk0 = rk0Init;
        rk1 = rk1Init;
        rk2 = rk2Init;
        rk3 = rk3Init;

        // Create plaintext as 32 bit unsigned integers
        s0 = pt4Init ^ s0 ^ rk0Init1;
        s1 = pt5Init ^ s1 ^ rk1Init1;
        s2 = pt6Init ^ s2 ^ rk2Init1;
        s3 = pt7Init ^ s3 ^ rk3Init1;

        // First round just XORs input with key.
        s0 = s0 ^ rk0;
        s1 = s1 ^ rk1;
        s2 = s2 ^ rk2;
        s3 = s3 ^ rk3;

        for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
            // Calculate round key
            u32 temp = rk3;
            rk0 = rk0 ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^ rconS[roundCount];
            rk1 = rk1 ^ rk0;
            rk2 = rk2 ^ rk1;
            rk3 = rk2 ^ rk3;
            // Table based round function
            t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk0;
            t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk1;
            t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk2;
            t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk3;
            s0 = t0;
            s1 = t1;
            s2 = t2;
            s3 = t3;
        }
        // Calculate the last round key
        temp = rk3;
        rk0 = rk0 ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^ rconS[ROUND_COUNT_MIN_1];
        // Last round uses s-box directly and XORs to produce output.
        s0 = arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rk0;
        if (s0 == ctS[0]) {
            rk1 = rk1 ^ rk0;
            s1 = arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rk1;
            if (s1 == ctS[1]) {
                rk2 = rk2 ^ rk1;
                s2 = arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rk2;
                if (s2 == ctS[2]) {
                    rk3 = rk2 ^ rk3;
                    s3 = arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rk3;
                    if (s3 == ctS[3]) {
                        printf("Message Block 2 Found: ");
                        print_plaintext(pt4Init);
                        print_plaintext(pt5Init);
                        printf("\n\n-------------------------------\n");
                    }
                }
            }
        }

        // Overflow
        if (index2 == MAX_4CHAR_VAL - 1) {
            index1++;
            index2 = 0;
            pt4Init = derive_plaintext(index1, t_charset);
        } else {
            index2++;
        }
        pt5Init = derive_plaintext(index2, t_charset);
    }
}

__host__ int CMAC128ExhaustiveSearch()
{
    printf("\n########## AES-CMAC-128 Exhaustive Search Implementation ##########\n\n");

    // Allocate 1st plaintext block, ciphertext, round key, k1
    u32 *pt, *ct, *rk, *rk1;
    gpuErrorCheck(cudaMallocManaged(&pt, 4 * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&ct, 4 * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&rk, 4 * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&rk1, 4 * sizeof(u32)));

    pt[0] = 0x58585858U;
    pt[1] = 0x58585858U;
    pt[2] = 0x58585858U;
    pt[3] = 0x58585858U;

    ct[0] = 0x37AC6997U;
    ct[1] = 0xA5E1C405U;
    ct[2] = 0x3FB5D625U;
    ct[3] = 0xAC598BF9U;

    rk[0] = 0x12345678U;
    rk[1] = 0x90ABCDEFU;
    rk[2] = 0x12345678U;
    rk[3] = 0x90ABCDEFU;

    rk1[0] = 0x026EBFD5U;
    rk1[1] = 0x5154D1EDU;
    rk1[2] = 0x852DC5BDU;
    rk1[3] = 0xF0DF1C0FU;

    u32* rcon;
    gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE * sizeof(u32)));
    for (int i = 0; i < RCON_SIZE; i++) {
        rcon[i] = RCON32[i];
    }
    // Allocate Tables
    u32 *t0, *t1, *t2, *t3, *t4, *t4_0, *t4_1, *t4_2, *t4_3;
    u8* SAES_d; // Cihangir
    u8* t_charset;
    gpuErrorCheck(cudaMallocManaged(&t0, TABLE_SIZE * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&t1, TABLE_SIZE * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&t2, TABLE_SIZE * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&t3, TABLE_SIZE * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&t4, TABLE_SIZE * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&t4_0, TABLE_SIZE * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&t4_1, TABLE_SIZE * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&t4_2, TABLE_SIZE * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&t4_3, TABLE_SIZE * sizeof(u32)));
    gpuErrorCheck(cudaMallocManaged(&SAES_d, 256 * sizeof(u8))); // Cihangir
    gpuErrorCheck(cudaMallocManaged(&t_charset, 36 * sizeof(u8)));
    for (int i = 0; i < TABLE_SIZE; i++) {
        t0[i] = T0[i];
        t1[i] = T1[i];
        t2[i] = T2[i];
        t3[i] = T3[i];
        t4[i] = T4[i];
        t4_0[i] = T4_0[i];
        t4_1[i] = T4_1[i];
        t4_2[i] = T4_2[i];
        t4_3[i] = T4_3[i];
    }
    for (int i = 0; i < 256; i++)
        SAES_d[i] = SAES[i]; // Cihangir
    for (int i = 0; i < 36; i++)
        t_charset[i] = T_CHARSET[i];

    printf("-------------------------------\n");
    u64* range = calculateCMACRange();

    clock_t beginTime = clock();

    blue_mesh_cmac_exhaustive_search<<<BLOCKS, THREADS>>>(pt, ct, rk, rk1, t0, t4, rcon, range, SAES_d, t_charset);


    cudaDeviceSynchronize();
    printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);

    printf("-------------------------------\n");
    printLastCUDAError();
    cudaFree(range);
    cudaFree(pt);
    cudaFree(ct);
    cudaFree(rk);
    cudaFree(rk1);
    cudaFree(t0);
    cudaFree(t1);
    cudaFree(t2);
    cudaFree(t3);
    cudaFree(t4);
    cudaFree(t4_0);
    cudaFree(t4_1);
    cudaFree(t4_2);
    cudaFree(t4_3);
    cudaFree(rcon);
    cudaFree(SAES_d);
    cudaFree(t_charset);
    return 0;
}
