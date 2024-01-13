from py_cuda_paillier import CudaConfig, PaillierPublicKey, PaillierPrivateKey, Homomorphic
from py_cuda_paillier import PaillierKeyPairGenerator as pkpg
import os
import time
import random
import argparse
from numba import cuda


'''
性能测试思路：
随机生成一个长度为 n 的明文消息列表
因为加密和解密的正确性已经经过验证了
所以这里分开测试 k 轮 encryption 和 k 轮 decryption
'''

'''
K:int
plain_text:list
public_key:PaillierPublicKey
mode:int (0: cpu-naive, 1: cpu-normal, 2: cuda-original, 3: cuda-plus, 4: cuda-test)
cuda_config:CudaConfig
'''
def timing_encryption(args):
    k = args.iter_num
    plain_text = args.plain_text
    public_key = args.public_key
    mode = args.mode
    cuda_config = args.cuda_config

    assert isinstance(k, int)
    assert isinstance(plain_text, list)
    assert isinstance(public_key, PaillierPublicKey)
    assert isinstance(mode, int)
    assert isinstance(cuda_config, CudaConfig)

    pure_sum = 0.0
    if mode==0:
        t1 = time.time()
        for i in range(k):
            encrypt_text, pure_time = public_key.encryption_naive(plain_text)
            pure_sum += pure_time
        t2 = time.time()
    elif mode==1:
        t1 = time.time()
        for i in range(k):
            encrypt_text, pure_time = public_key.encryption(plain_text)
            pure_sum += pure_time
        t2 = time.time()
    elif mode==2:
        t1 = time.time()
        for i in range(k):
            encrypt_text, pure_time = public_key.cuda_encryption(plain_text, cuda_config)
            pure_sum += pure_time
        t2 = time.time()
    elif mode==3:
        t1 = time.time()
        for i in range(k):
            encrypt_text, pure_time = public_key.cuda_plus_encryption(plain_text, cuda_config)
            pure_sum += pure_time
        t2 = time.time()
        
    duration = t2 - t1
    return duration, pure_sum, encrypt_text

def timing_decryption(args):
    k = args.iter_num
    cipher_text = args.cipher_text
    private_key = args.private_key
    mode = args.mode
    cuda_config = args.cuda_config
    # 在这里断言一下变量类型，后面的代码中解释器应该就不用猜变量类型了
    assert isinstance(k, int)
    assert isinstance(cipher_text, list)
    assert isinstance(private_key, PaillierPrivateKey)
    assert isinstance(mode, int)
    assert isinstance(cuda_config, CudaConfig)
    pure_sum = 0.0
    if mode==0:
        t1 = time.time()
        for i in range(k):
            decrypt_text, pure_time = private_key.decryption_naive(cipher_text)
            pure_sum += pure_time
        t2 = time.time()
    elif mode==1:
        t1 = time.time()
        for i in range(k):
            decrypt_text, pure_time = private_key.decryption(cipher_text)
            pure_sum += pure_time
        t2 = time.time()
    elif mode==2:
        t1 = time.time()
        for i in range(k):
            decrypt_text, pure_time = private_key.cuda_decryption(cipher_text, cuda_config)
            pure_sum += pure_time
        t2 = time.time()
    elif mode==3:
        t1 = time.time()
        for i in range(k):
            decrypt_text, pure_time = private_key.cuda_plus_decryption(cipher_text, cuda_config)
            pure_sum += pure_time
        t2 = time.time()
    
    duration = t2 - t1
    return duration, pure_sum, decrypt_text

def check_correctness(list0:list, list1:list):
    result = True
    assert len(list0) == len(list1)
    for i in range(len(list0)):
        elem0 = list0[i]
        elem1 = list1[i]
        if elem0 == elem1:
            continue
        else:
            result = False
            print("Wrong encryption/decryption!")
            print(elem0,elem1)
            break
    if result:
        print("Correct encryption/decryption!")
    return result


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_num", help="number of iterations", type=int, default=10)
    parser.add_argument("--bit_key_length", help="key length in bits", type=int, default=32)
    parser.add_argument("--threads_per_block", help="threads per block", type=int, default=128)
    parser.add_argument("--blocks_per_grid", help="blocks per grid", type=int, default=128)
    parser.add_argument("--text_length", help="length of plain/cipher text", type=int, default=128*128)
    parser.add_argument("--scalar", help="scalar used for homomorphic scalar multiplication", type=int, default=1234)
    args = parser.parse_args()
    args.cuda_config = CudaConfig(threads_per_block=args.threads_per_block,blocks=args.blocks_per_grid)
    if args.text_length != args.threads_per_block * args.blocks_per_grid:
        args.text_length = args.threads_per_block * args.blocks_per_grid
    public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(p=709,q=919)
    args.public_key = public_key
    args.private_key = private_key
    # plain text for encryption
    args.plain_text = [random.randrange(100, public_key.n) for i in range(args.text_length)]
    args.cipher_text, _ = args.public_key.cuda_encryption(args.plain_text, args.cuda_config)
    '''
    mode = 0: cpu-naive
    mode = 1: cpu-normal
    mode = 2: cuda-original
    mode = 3: cuda-plus
    mode = 4: cuda-test
    '''
    # result_{mode}_{operation}
    args.mode = 0
    print("using mode = 0...")
    time_0_0, pure_0_0, result_0_0 = timing_encryption(args)
    print("0_0 timing_encryption\t{:.6f}\t{:.6f}".format(time_0_0/args.iter_num, pure_0_0/args.iter_num))
    args.cipher_text = result_0_0
    time_0_1, pure_0_1, result_0_1 = timing_decryption(args)
    print("0_1 timing_decryption\t{:.6f}\t{:.6f}".format(time_0_1/args.iter_num, pure_0_1/args.iter_num))
    check_correctness(args.plain_text, result_0_1)

    args.mode = 1
    print("using mode = 1...")
    time_1_0, pure_1_0, result_1_0 = timing_encryption(args)
    print("1_0 timing_encryption\t{:.6f}\t{:.6f}".format(time_1_0/args.iter_num, pure_1_0/args.iter_num))
    args.cipher_text = result_1_0
    time_1_1, pure_1_1, result_1_1 = timing_decryption(args)
    print("1_1 timing_decryption\t{:.6f}\t{:.6f}".format(time_1_1/args.iter_num, pure_1_1/args.iter_num))
    check_correctness(args.plain_text, result_1_1)

    args.mode = 2
    print("using mode = 2...")
    time_2_0, pure_2_0, result_2_0 = timing_encryption(args)
    print("2_0 timing_encryption\t{:.6f}\t{:.6f}".format(time_2_0/args.iter_num, pure_2_0/args.iter_num))
    args.cipher_text = result_2_0
    time_2_1, pure_2_1, result_2_1 = timing_decryption(args)
    print("2_1 timing_decryption\t{:.6f}\t{:.6f}".format(time_2_1/args.iter_num, pure_2_1/args.iter_num))
    check_correctness(args.plain_text, result_2_1)

    args.mode = 3
    print("using mode = 3...")
    time_3_0, pure_3_0, result_3_0 = timing_encryption(args)
    print("3_0 timing_encryption\t{:.6f}\t{:.6f}".format(time_3_0/args.iter_num, pure_3_0/args.iter_num))
    args.cipher_text = result_3_0
    time_3_1, pure_3_1, result_3_1 = timing_decryption(args)
    print("3_1 timing_decryption\t{:.6f}\t{:.6f}".format(time_3_1/args.iter_num, pure_3_1/args.iter_num))
    check_correctness(args.plain_text, result_3_1)

    return
    
if __name__ == "__main__":
    main()





